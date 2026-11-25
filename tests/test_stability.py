import cv2
import numpy as np
import pytest

from welog.controller import wait_for_stable
from welog.demo import make_scene
from welog.models import AlignmentConfig, CaptureConfig
from welog.recognition import achromatic_media_regions
from welog.stability import LayoutStabilityDetector


def test_local_animation_is_tolerated_only_with_stationary_content():
    base = make_scene(720, 640, seed=111)
    detector = LayoutStabilityDetector(AlignmentConfig())
    rng = np.random.default_rng(9)
    results = []
    for _ in range(3):
        sample = base.copy()
        sample[20:110, 490:620] = rng.integers(0, 256, (90, 130, 3), dtype=np.uint8)
        results.append(detector.observe(sample))
    result = results[-1]
    assert result.stable, result.to_dict()
    assert result.reason == "local_animation"
    assert 0 < result.dynamic_region_ratio <= 0.12
    assert result.static_matches >= 3
    assert result.dynamic_masks


def test_slow_global_scrolling_is_not_hidden_as_animation():
    scene = make_scene(900, 640, seed=111)
    detector = LayoutStabilityDetector(AlignmentConfig())
    assert all(not detector.observe(scene[y:y + 720]).stable for y in range(15))


def test_sparse_moving_text_cannot_be_masked_away():
    scene = np.full((900, 640, 3), 242, np.uint8)
    cv2.putText(scene, "a single moving message", (75, 360), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (30, 30, 30), 1, cv2.LINE_AA)
    detector = LayoutStabilityDetector(AlignmentConfig())
    assert all(not detector.observe(scene[y:y + 720]).stable for y in range(10))


def test_inertia_must_end_before_three_consistent_samples_are_accepted():
    scene = make_scene(1000, 640)
    detector = LayoutStabilityDetector(AlignmentConfig())
    offsets = [0, 30, 47, 56, 56, 56]
    results = [detector.observe(scene[y:y + 720]) for y in offsets]
    assert not any(result.stable for result in results[:-1])
    assert results[-1].stable


def test_widespread_animation_and_layout_changes_remain_unstable():
    detector = LayoutStabilityDetector(AlignmentConfig())
    rng = np.random.default_rng(12)
    for _ in range(5):
        sample = rng.integers(0, 256, (720, 640, 3), dtype=np.uint8)
        assert not detector.observe(sample).stable


def test_wait_returns_animation_masks_without_waiting_for_every_pixel_to_stop():
    base = make_scene(720, 640)
    rng = np.random.default_rng(31)

    class AnimatedSource:
        def read(self):
            sample = base.copy()
            sample[20:110, 490:620] = rng.integers(0, 256, (90, 130, 3), dtype=np.uint8)
            return sample

    diagnostics = {}
    result = wait_for_stable(AnimatedSource(), CaptureConfig(sample_interval_s=0.001),
                             AlignmentConfig(), diagnostics=diagnostics)
    assert result.shape == base.shape
    assert diagnostics["stable"]
    assert diagnostics["dynamic_masks"]


def test_repeated_media_with_tiny_clipped_border_noise_is_stationary():
    rng = np.random.default_rng(18653)
    scene = np.full((663, 880, 3), 247, np.uint8)
    picture = rng.integers(10, 230, (132, 132, 3), dtype=np.uint8)
    avatar = rng.integers(10, 230, (36, 36, 3), dtype=np.uint8)
    for y in (77, 229, 381):
        scene[y:y+132, 687:819] = picture
        scene[y:y+36, 833:869] = avatar
    detector = LayoutStabilityDetector(AlignmentConfig())
    for i in range(3):
        sample = scene.copy()
        sample[0, 707+i:720+i] -= i*2
        result = detector.observe(sample)
    assert result.stable and result.reason == 'minor_border_noise'
    assert not result.dynamic_masks


def test_border_rule_does_not_swallow_low_contrast_messages_moving_interior():
    scene = np.full((750, 640, 3), 247, np.uint8)
    cv2.putText(scene, 'moving low contrast text', (70, 300), cv2.FONT_HERSHEY_SIMPLEX, .5, (243,)*3, 1)
    detector = LayoutStabilityDetector(AlignmentConfig())
    for i in range(4):
        sample = scene[i:i+663].copy()
        sample[0, 67:80] -= i
        result = detector.observe(sample)
    assert not result.stable


@pytest.mark.parametrize('grayscale', [False, True])
def test_dense_grayscale_animation_clipped_at_viewport_edges_is_local(grayscale):
    rng = np.random.default_rng(33109)
    height, width = 663, 880
    scene = np.full((height+12, width), 247, np.uint8)
    graphics = [(0, 77), (153, 285), (361, 493), (569, height+12)]
    avatar = rng.integers(0, 210, (36, 36), dtype=np.uint8)
    for index, y in enumerate((97, 305, 513)):
        scene[y:y+36, 20:56] = avatar
        cv2.rectangle(scene, (65, y), (232, y+35), 235, -1)
        cv2.putText(scene, f'Fixed message {index}', (81, y+24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, 30, 1, cv2.LINE_AA)

    def sample(offset):
        image = scene.copy()
        for top, bottom in graphics:
            image[top:bottom, 70:192] = rng.integers(20, 205, (bottom-top, 122), dtype=np.uint8)
        viewport = image[offset:offset+height]
        return viewport if grayscale else cv2.cvtColor(viewport, cv2.COLOR_GRAY2BGR)

    stationary = LayoutStabilityDetector(AlignmentConfig())
    moving = LayoutStabilityDetector(AlignmentConfig())
    for offset in (0, 4, 8):
        result = stationary.observe(sample(0))
        assert not moving.observe(sample(offset)).stable
    assert result.stable, result.to_dict()
    assert result.dynamic_masks and result.outside_media_pixels == 0
    assert result.media_supported_ratio == 1


def test_dense_gray_flat_panel_with_sparse_text_is_not_media():
    image = np.full((300, 400), 247, np.uint8)
    image[225:, 65:195] = 130
    cv2.putText(image, 'simple text', (76, 265), cv2.FONT_HERSHEY_SIMPLEX, .5, 40, 1)
    assert not achromatic_media_regions(image, include_small_edge_media=True)

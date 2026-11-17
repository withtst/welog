"""Read supplied offline images into RAM and write only a text validation report."""
import argparse
import json
from pathlib import Path

from welog.file_cards import mark_candidates
from welog.imageio import read_image
from welog.pipeline import FrameProcessor
from welog.recognition import RapidOcrAdapter,assign_lines,detect_bubbles
from welog.timeline import parse_timestamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    backend=RapidOcrAdapter()
    results=[]
    for path in args.images:
        image=read_image(path)
        bubbles=detect_bubbles(image)
        mark_candidates(image,bubbles)
        clean=image.copy()
        FrameProcessor.mask_media(clean,[bubble.box for bubble in bubbles if bubble.kind!='text' and '_file_candidate' not in bubble.metadata])
        labels=assign_lines(bubbles,backend.recognize(clean))
        results.append({'source':path.name,'files':[bubble.metadata['file'] for bubble in bubbles if bubble.kind=='file'],
                        'timestamps':[{'raw':line.text,'parsed':parse_timestamp(line.text)} for line in labels if parse_timestamp(line.text)]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'desktop_accessed':False,'images_written':0,'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'files':[item['filename'] for row in results for item in row['files']],
                      'timestamps':sum(len(row['timestamps']) for row in results),'images_written':0},ensure_ascii=False))


if __name__=='__main__':main()

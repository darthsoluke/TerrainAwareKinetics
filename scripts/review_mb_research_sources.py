"""Retrieve primary research paragraphs for the online reference review."""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path

import requests


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ''
        self.in_title = False
        self.paragraph = None
        self.items = []

    def handle_starttag(self,tag,attrs):
        if tag=='title': self.in_title=True
        if tag=='p' and 'ltx_p' in dict(attrs).get('class','').split(): self.paragraph=[]

    def handle_data(self,data):
        if self.in_title: self.title+=data
        if self.paragraph is not None: self.paragraph.append(data)

    def handle_endtag(self,tag):
        if tag=='title': self.in_title=False
        if tag=='p' and self.paragraph is not None:
            self.items.append(' '.join(''.join(self.paragraph).split()))
            self.paragraph=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--terms',nargs='+',required=True)
    args=parser.parse_args()
    response=requests.get(args.url,timeout=40)
    response.raise_for_status()
    html=Paragraphs(); html.feed(response.content.decode('utf-8'))
    result=dict(url=args.url,title=html.title,paragraph_count=len(html.items),
        matches=[dict(index=i,text=p) for i,p in enumerate(html.items)
                 if any(term.lower() in p.lower() for term in args.terms)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(url=result['url'],title=result['title'],paragraphs=result['paragraph_count'],matches=result['matches'][:7]),ensure_ascii=True))


if __name__=='__main__': main()

# _*_ coding:utf-8 _*_
import glob
import json
datas = glob.glob("chinese-poetry/全唐诗/poet*json")# 使用通配符的方式来加载整个json文件
for data in datas:
    with open(data,'r',encoding='utf-8') as fp:
        tangshi = json.load(fp)
        for poem in tangshi:
            if len(poem["paragraphs"]) == 2 and len(poem["paragraphs"][0]) == 12 and len(poem["paragraphs"][1]) == 12:
                with open("data/gushi/gushi.txt","a",encoding = 'utf-8') as f:
                    f.write("".join(poem['paragraphs']))
                    f.write("\n")

from datasets import load_dataset

ds = load_dataset("breadlicker45/youtube-comments-v2")
# ds = load_dataset("Cropinky/rap_lyrics_english")     


f = open('input2.txt', 'w', encoding='utf-8')
text = ''.join(' ' + x for x in ds['train']['text'])
f.write(text)
f.close()
# print(ds['train']['text'])

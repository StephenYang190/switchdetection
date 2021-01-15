import os


files = os.listdir('./data/')

with open('train.lst', 'w') as f:
    for file in files:
        dirname = './data/' + file
        if file == 'open':
            label = 1
        else:
            label = 0
        imgs = os.listdir(dirname)
        for img in imgs:
            if '.png' in img:
                f.write(dirname + '/' + img + ' %d' % label)
                f.write('\n')
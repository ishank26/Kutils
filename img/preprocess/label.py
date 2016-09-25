import glob
import data_Augment as da
norm= glob.glob("/home/neo/work/cnn_down/data/baby_box/*nor*")
down = glob.glob("/home/neo/work/cnn_down/data/baby_box/*down*")


print len(norm)
print len(down)
#train_nor=nor[:int(len(nor)*0.6+1)] 
#test_nor=nor[int(len(nor)*0.6+2):int(len(nor)*0.6+2)+int(len(nor)*0.2+1)]
#val_nor=nor[int(len(nor)*0.6+2)+int(len(nor)*0.2+2):]

#down=da.get_data(down)
#translated_d=da.translate(down)
#rotate_d=da.rotate(down)
#da.out_img(translated_d, rotate_d, "b_down")


#norm=da.get_data(norm)
#translated_n=da.translate(norm)
#rotate_n=da.rotate(norm)
#da.out_img(translated_n, rotate_n, "b_nor")




#print len(train_nor) ,train_nor[-1:]
#print len(test_nor), test_nor[0], test_nor[-1:]
#print len(val_nor), val_nor[0]
labels= open("/home/neo/work/cnn_down/data/224_align_col/labels.txt", "a")

for i in norm:
	labels.write(i+" 0\n")

for i in down:
	labels.write(i+" 1\n")

file.close(labels)

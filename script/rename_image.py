import os

path = 'home/usr/images/animals' #Change to your image path

imagelist = os.listdir(path)

count = 1
for image in imagelist:
    new_name = 'wolf_image_{}'.format(str(count))
    os.rename((path+'/'+image),(path+'/'+new_name))
    count = count+1

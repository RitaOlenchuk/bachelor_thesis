echo "labels"

for file in `ls /usr/local/hdd/tfunet/aortack/sequences/train/labels_plaque/*.tiff`;
do

    if [[ $file == *"small"* ]]; then
        continue
    fi
    convert $file -resize 640x480! $file.small.tif;

done

echo "regular"

for file in `ls /usr/local/hdd/tfunet/aortack/sequences/train/*.tif`;
do

    if [[ $file == *"small"* ]]; then
        continue
    fi
    convert $file -resize 640x480! $file.small.tif;

done

echo 'test'
for file in `ls /usr/local/hdd/tfunet/aortack/sequences/test/*.tif`;
do

    if [[ $file == *"small"* ]]; then
        continue
    fi
    convert $file -resize 640x480! $file.small.tif;

done
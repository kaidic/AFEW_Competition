for folder in 'Angry' 'Disgust' 'Fear' 'Happy' 'Neutral' 'Sad' 'Surprise'
#for folder in 'Angry'
	do
		for video in `find $folder/*.avi`
		do 

			NAME=`echo $video | cut -d "/" -f2`;
			NAME=`echo $NAME | cut -d "." -f1`;
			echo "$NAME"
			ffmpeg -i "$video" -vn -b:a 128k -f wav "$folder/$NAME"".wav"
			SMILExtract_Release -C emobase2010.conf -I "$folder/$NAME"".wav" -O "$folder/$NAME"".txt"
		done
done

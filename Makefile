main:
	mkdir submission
	cp classifier submission
	cp models submission
	cp superres submission
	cp superres_utils submission
	cp *.txt submission
	cp *.py submission
	cp utkface.sh submission
	rm -fr submission/**/__pycache__
	rm submission/classifier/bins_f.txt
	zip -r submission.zip
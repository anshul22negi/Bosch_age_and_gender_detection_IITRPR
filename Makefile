main:
	mkdir submission
	cp -r classifier submission
	cp -r models submission
	cp -r superres submission
	cp -r superres_utils submission
	cp *.txt submission
	cp *.py submission
	cp utkface.sh submission
	rm -fr submission/**/__pycache__
	rm submission/classifier/bins_f.txt
	cp README.pdf submission/MP_BO_T19_UM.pdf
	mv submission MP_BO_T19_CODE
	zip -r MP_BO_T19_CODE.zip MP_BO_T19_CODE
	rm -fr MP_BO_T19_CODE
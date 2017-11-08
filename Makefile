default:
	nvcc -arch=sm_52 -o match match.cu
clean:
	rm -f match

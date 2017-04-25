main.o: main.cpp Occupancy.cpp Occupancy.h Occupancy_GPU.cu
	nvcc -std=c++11 -w  -O3  main.cpp  Occupancy.cpp Occupancy_GPU.cu `pkg-config --cflags --libs opencv`   -o main.o
cpu: main.cpp Occupancy.cpp Occupancy.h 
	g++ -std=c++11 -w  -O3  main.cpp  Occupancy.cpp  `pkg-config --cflags --libs opencv`   -o main_cpu.o
gpu: main.cpp  Occupancy.h Occupancy_GPU.cu	
	nvcc -std=c++11 -w  -O3  main.cpp  Occupancy_GPU.cu `pkg-config --cflags --libs opencv`   -o main_gpu.o
clean:
	rm -f main.o main_cpu.o main_gpu.o
run: 
	time ./main.o
run_cpu: 
	time ./main_cpu.o
run_gpu: 
	time ./main_gpu.o

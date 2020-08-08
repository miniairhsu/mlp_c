incs=-I/home/swlib/gsl/include -I.	
libs=-L/home/swlib/gsl/lib -L.
ldflags=-lgsl -lgslcblas -lm 
deps := $(wildcard *.h) 

AP: main.o mlp.o weight_table.o
	gcc main.o mlp.o weight_table.o $(ldflags) $(libs) $(incs) -o main.exe

main.o:main.c $(deps)
	gcc -c main.c $(ldflags) $(libs) $(incs) -o main.o  


mlp.o:mlp.c $(deps)
	gcc -c mlp.c $(ldflags) $(libs) $(incs) -o mlp.o

weight_table.o:weight_table.c $(deps)
	gcc -c weight_table.c $(ldflags) $(libs) $(incs) -o weight_table.o 

clean:
	rm -f *.o .logibone_fifo.ko.cmd .logibone_fifo.mod.o.cmd .logibone_fifo.o.cmd
	rm -f *.ko .tmp_versions/*
	rm -f logibone_fifo.mod.*
	rm -f [mM]odule*
	rm -f *.exe
	rmdir .tmp_versions/




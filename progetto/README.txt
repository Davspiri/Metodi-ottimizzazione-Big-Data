INPUT FILE

The MOBD simulator expects a text file containing the positions of the modules x^1,...,x^m.
The input file must contain exactly 2000 lines, with one real number per line, no empty lines, and no additional characters.
Each line of the input file contains a coordinate of the 1000 modules in the following order:

(x^1)_1
(x^1)_2
(x^2)_1
(x^2)_2
...
(x^1000)_1
(x^1000)_2

USAGE

From a command prompt or terminal:

mobd file_name -b
mobd file_name -t

For example, you can test the simulator with the file 'x.txt' provided in the current directory.

OPTIONS

-b: evaluates only the black-box cost component of the objective function, printing a single numerical value.

-t: computes the total objective function (this is provided only for verification and debugging).

OUTPUT

For option -b: a single number rounded to two decimal digits.
For option -t: the total cost followed by the individual components of the objective function.
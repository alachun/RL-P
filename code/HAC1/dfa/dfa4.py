dfa_text = '''
0 -> 0 [label="~a & ~b & ~c & ~d & ~e"];
0 -> 1 [label="e & ~a & ~b & ~c & ~d & ~f"];
0 -> 2 [label="e & f & ~a & ~b & ~c & ~d & ~g"];
0 -> 3 [label="e & f & g & ~a & ~b & ~c & ~d & ~h"];
0 -> 4 [label="e & f & g & h & ~a & ~b & ~c & ~d"];
0 -> 24 [label="b | c | d"];
0 -> 5 [label="a & ~b & ~c & ~d & ~e"];
0 -> 6 [label="a & e & ~b & ~c & ~d & ~f"];
0 -> 7 [label="a & e & f & ~b & ~c & ~d & ~g"];
0 -> 8 [label="a & e & f & g & ~b & ~c & ~d & ~h"];
0 -> 9 [label="a & e & f & g & h & ~b & ~c & ~d"];
1 -> 1 [label="~a & ~b & ~c & ~d & ~f"];
1 -> 2 [label="f & ~a & ~b & ~c & ~d & ~g"];
1 -> 3 [label="f & g & ~a & ~b & ~c & ~d & ~h"];
1 -> 4 [label="f & g & h & ~a & ~b & ~c & ~d"];
1 -> 24 [label="b | c | d"];
1 -> 6 [label="a & ~b & ~c & ~d & ~f"];
1 -> 7 [label="a & f & ~b & ~c & ~d & ~g"];
1 -> 8 [label="a & f & g & ~b & ~c & ~d & ~h"];
1 -> 9 [label="a & f & g & h & ~b & ~c & ~d"];
2 -> 2 [label="~a & ~b & ~c & ~d & ~g"];
2 -> 3 [label="g & ~a & ~b & ~c & ~d & ~h"];
2 -> 4 [label="g & h & ~a & ~b & ~c & ~d"];
2 -> 24 [label="b | c | d"];
2 -> 7 [label="a & ~b & ~c & ~d & ~g"];
2 -> 8 [label="a & g & ~b & ~c & ~d & ~h"];
2 -> 9 [label="a & g & h & ~b & ~c & ~d"];
3 -> 3 [label="~a & ~b & ~c & ~d & ~h"];
3 -> 4 [label="h & ~a & ~b & ~c & ~d"];
3 -> 24 [label="b | c | d"];
3 -> 8 [label="a & ~b & ~c & ~d & ~h"];
3 -> 9 [label="a & h & ~b & ~c & ~d"];
4 -> 4 [label="~a & ~b & ~c & ~d"];
4 -> 24 [label="b | c | d"];
4 -> 9 [label="a & ~b & ~c & ~d"];
24 -> 24 [label="True"];
5 -> 24 [label="c | d | ~a"];
5 -> 5 [label="a & ~b & ~c & ~d & ~e"];
5 -> 6 [label="a & e & ~b & ~c & ~d & ~f"];
5 -> 7 [label="a & e & f & ~b & ~c & ~d & ~g"];
5 -> 8 [label="a & e & f & g & ~b & ~c & ~d & ~h"];
5 -> 9 [label="a & e & f & g & h & ~b & ~c & ~d"];
5 -> 10 [label="a & b & ~c & ~d & ~e"];
5 -> 11 [label="a & b & e & ~c & ~d & ~f"];
5 -> 12 [label="a & b & e & f & ~c & ~d & ~g"];
5 -> 13 [label="a & b & e & f & g & ~c & ~d & ~h"];
5 -> 14 [label="a & b & e & f & g & h & ~c & ~d"];
6 -> 24 [label="c | d | ~a"];
6 -> 6 [label="a & ~b & ~c & ~d & ~f"];
6 -> 7 [label="a & f & ~b & ~c & ~d & ~g"];
6 -> 8 [label="a & f & g & ~b & ~c & ~d & ~h"];
6 -> 9 [label="a & f & g & h & ~b & ~c & ~d"];
6 -> 11 [label="a & b & ~c & ~d & ~f"];
6 -> 12 [label="a & b & f & ~c & ~d & ~g"];
6 -> 13 [label="a & b & f & g & ~c & ~d & ~h"];
6 -> 14 [label="a & b & f & g & h & ~c & ~d"];
7 -> 24 [label="c | d | ~a"];
7 -> 7 [label="a & ~b & ~c & ~d & ~g"];
7 -> 8 [label="a & g & ~b & ~c & ~d & ~h"];
7 -> 9 [label="a & g & h & ~b & ~c & ~d"];
7 -> 12 [label="a & b & ~c & ~d & ~g"];
7 -> 13 [label="a & b & g & ~c & ~d & ~h"];
7 -> 14 [label="a & b & g & h & ~c & ~d"];
8 -> 24 [label="c | d | ~a"];
8 -> 8 [label="a & ~b & ~c & ~d & ~h"];
8 -> 9 [label="a & h & ~b & ~c & ~d"];
8 -> 13 [label="a & b & ~c & ~d & ~h"];
8 -> 14 [label="a & b & h & ~c & ~d"];
9 -> 24 [label="c | d | ~a"];
9 -> 9 [label="a & ~b & ~c & ~d"];
9 -> 14 [label="a & b & ~c & ~d"];
10 -> 24 [label="d | ~a | ~b"];
10 -> 10 [label="a & b & ~c & ~d & ~e"];
10 -> 11 [label="a & b & e & ~c & ~d & ~f"];
10 -> 12 [label="a & b & e & f & ~c & ~d & ~g"];
10 -> 13 [label="a & b & e & f & g & ~c & ~d & ~h"];
10 -> 14 [label="a & b & e & f & g & h & ~c & ~d"];
10 -> 15 [label="a & b & c & ~d & ~e"];
10 -> 16 [label="a & b & c & e & ~d & ~f"];
10 -> 17 [label="a & b & c & e & f & ~d & ~g"];
10 -> 18 [label="a & b & c & e & f & g & ~d & ~h"];
10 -> 19 [label="a & b & c & e & f & g & h & ~d"];
11 -> 24 [label="d | ~a | ~b"];
11 -> 11 [label="a & b & ~c & ~d & ~f"];
11 -> 12 [label="a & b & f & ~c & ~d & ~g"];
11 -> 13 [label="a & b & f & g & ~c & ~d & ~h"];
11 -> 14 [label="a & b & f & g & h & ~c & ~d"];
11 -> 16 [label="a & b & c & ~d & ~f"];
11 -> 17 [label="a & b & c & f & ~d & ~g"];
11 -> 18 [label="a & b & c & f & g & ~d & ~h"];
11 -> 19 [label="a & b & c & f & g & h & ~d"];
12 -> 24 [label="d | ~a | ~b"];
12 -> 12 [label="a & b & ~c & ~d & ~g"];
12 -> 13 [label="a & b & g & ~c & ~d & ~h"];
12 -> 14 [label="a & b & g & h & ~c & ~d"];
12 -> 17 [label="a & b & c & ~d & ~g"];
12 -> 18 [label="a & b & c & g & ~d & ~h"];
12 -> 19 [label="a & b & c & g & h & ~d"];
13 -> 24 [label="d | ~a | ~b"];
13 -> 13 [label="a & b & ~c & ~d & ~h"];
13 -> 14 [label="a & b & h & ~c & ~d"];
13 -> 18 [label="a & b & c & ~d & ~h"];
13 -> 19 [label="a & b & c & h & ~d"];
14 -> 24 [label="d | ~a | ~b"];
14 -> 14 [label="a & b & ~c & ~d"];
14 -> 19 [label="a & b & c & ~d"];
15 -> 24 [label="~a | ~b | ~c"];
15 -> 15 [label="a & b & c & ~d & ~e"];
15 -> 16 [label="a & b & c & e & ~d & ~f"];
15 -> 17 [label="a & b & c & e & f & ~d & ~g"];
15 -> 18 [label="a & b & c & e & f & g & ~d & ~h"];
15 -> 19 [label="a & b & c & e & f & g & h & ~d"];
15 -> 20 [label="a & b & c & d & ~e"];
15 -> 21 [label="a & b & c & d & e & ~f"];
15 -> 22 [label="a & b & c & d & e & f & ~g"];
15 -> 23 [label="a & b & c & d & e & f & g & ~h"];
15 -> 25 [label="a & b & c & d & e & f & g & h"];
16 -> 24 [label="~a | ~b | ~c"];
16 -> 16 [label="a & b & c & ~d & ~f"];
16 -> 17 [label="a & b & c & f & ~d & ~g"];
16 -> 18 [label="a & b & c & f & g & ~d & ~h"];
16 -> 19 [label="a & b & c & f & g & h & ~d"];
16 -> 21 [label="a & b & c & d & ~f"];
16 -> 22 [label="a & b & c & d & f & ~g"];
16 -> 23 [label="a & b & c & d & f & g & ~h"];
16 -> 25 [label="a & b & c & d & f & g & h"];
17 -> 24 [label="~a | ~b | ~c"];
17 -> 17 [label="a & b & c & ~d & ~g"];
17 -> 18 [label="a & b & c & g & ~d & ~h"];
17 -> 19 [label="a & b & c & g & h & ~d"];
17 -> 22 [label="a & b & c & d & ~g"];
17 -> 23 [label="a & b & c & d & g & ~h"];
17 -> 25 [label="a & b & c & d & g & h"];
18 -> 24 [label="~a | ~b | ~c"];
18 -> 18 [label="a & b & c & ~d & ~h"];
18 -> 19 [label="a & b & c & h & ~d"];
18 -> 23 [label="a & b & c & d & ~h"];
18 -> 25 [label="a & b & c & d & h"];
19 -> 24 [label="~a | ~b | ~c"];
19 -> 19 [label="a & b & c & ~d"];
19 -> 25 [label="a & b & c & d"];
20 -> 24 [label="~a | ~b | ~c | ~d"];
20 -> 20 [label="a & b & c & d & ~e"];
20 -> 21 [label="a & b & c & d & e & ~f"];
20 -> 22 [label="a & b & c & d & e & f & ~g"];
20 -> 23 [label="a & b & c & d & e & f & g & ~h"];
20 -> 25 [label="a & b & c & d & e & f & g & h"];
21 -> 24 [label="~a | ~b | ~c | ~d"];
21 -> 21 [label="a & b & c & d & ~f"];
21 -> 22 [label="a & b & c & d & f & ~g"];
21 -> 23 [label="a & b & c & d & f & g & ~h"];
21 -> 25 [label="a & b & c & d & f & g & h"];
22 -> 24 [label="~a | ~b | ~c | ~d"];
22 -> 22 [label="a & b & c & d & ~g"];
22 -> 23 [label="a & b & c & d & g & ~h"];
22 -> 25 [label="a & b & c & d & g & h"];
23 -> 24 [label="~a | ~b | ~c | ~d"];
23 -> 23 [label="a & b & c & d & ~h"];
23 -> 25 [label="a & b & c & d & h"];
25 -> 24 [label="~a | ~b | ~c | ~d"];
25 -> 25 [label="a & b & c & d"];'''

accepting_state = '25'
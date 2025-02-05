dfa_text = '''
0 -> 0 [label="~c1 & ~c2 & ~c3"];
0 -> 3 [label="c2 | c3"];
0 -> 1 [label="c1 & ~c2 & ~c3"];
3 -> 3 [label="True"];
1 -> 3 [label="c3 | ~c1"];
1 -> 1 [label="c1 & ~c2 & ~c3"];
1 -> 2 [label="c1 & c2 & ~c3"];
2 -> 3 [label="~c1 | ~c2"];
2 -> 2 [label="c1 & c2 & ~c3"];
2 -> 4 [label="c1 & c2 & c3"];
4 -> 3 [label="~c1 | ~c2 | ~c3"];
4 -> 4 [label="c1 & c2 & c3"];'''

accepting_state = '4'
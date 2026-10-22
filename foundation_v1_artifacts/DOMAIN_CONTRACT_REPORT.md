# Foundation V1 variable-J domain audit

Derived using the unchanged current exact-plan helper at both server counts.

|J|N|Requested lower|Requested upper|Effective lower|Effective upper|Equal legal|
|---|---|---|---|---|---|---|
|2|250|0.3|0.9|0.3|0.7|True|
|2|1000|0.3|0.9|0.3|0.7|True|
|3|250|0.19999999999999998|0.6|0.19999999999999998|0.6|True|
|3|1000|0.19999999999999998|0.6|0.19999999999999998|0.6|True|
|4|250|0.15|0.45|0.15|0.45|True|
|4|1000|0.15|0.45|0.15|0.45|True|
|5|250|0.12|0.36000000000000004|0.12|0.36000000000000004|True|
|5|1000|0.12|0.36000000000000004|0.12|0.36000000000000004|True|
|6|250|0.09999999999999999|0.3|0.09999999999999999|0.3|True|
|6|1000|0.09999999999999999|0.3|0.09999999999999999|0.3|True|

J2 requested upper.9 narrows to.7 due to the bounded simplex. Other effective intervals:J3 [.2,.6],J4 [.15,.45],J5 [.12,.36],J6 [.1,.3]. Every proposal is validated;equal and face/corner examples are in the CSV. Pbar/R are kW/server,from each actual workload: Pbar=[.9Pmin,Pmax],R≥.01,R≤.6Pbar,Pbar+R≤1.2Pmax,Pbar−R≥0. Unit sum tolerance1e-9;inherited P/R boundary tolerance1e-9. Input legality is separate from strict numerical feasibility and required evidence.

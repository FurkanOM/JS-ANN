# JS-ANN

Basic, lightweight, zero dependent, supervised, online learning tool.
It can be used for any purpose.

## Usage

```js
var network = new Network(trainData, i+1, 1, 0.0001, 1500, 0.15);
network.init();
// If you don't set momentum and weight decay values network automatically ignore these processes
network.setWeightDecay(0.001);
network.setMomentum(0.1);
// Training process
network.train();
// Test process
network.setTestData(testData);
network.test();
```

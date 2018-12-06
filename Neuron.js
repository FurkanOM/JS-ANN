function Neuron(){
	this.output;
	this.actual;
	this.error;
	this.weights;
	this.prevDeltas;
}

Neuron.prototype.initNeuron = function(connectionNumber){
	var weights = [];
	var deltas = [];
	for(var j = 0; j < connectionNumber; j++){
		weights.push(this.getRandomFloatInRange(0.01, -0.01));
		deltas.push(0);
	}
	this.setWeights(weights);
	this.setPrevDeltas(deltas);
};

Neuron.prototype.getRandomFloatInRange = function(max, min){
	return Math.random() * (max - min) + min;
};

Neuron.prototype.setOutput = function(output){
	this.output = output;
};

Neuron.prototype.getOutput = function(){
	return this.output;
};

Neuron.prototype.setActual = function(actual){
	this.actual = actual;
};

Neuron.prototype.getActual = function(){
	return this.actual;
};

Neuron.prototype.setError = function(error){
	this.error = error;
};

Neuron.prototype.getError = function(){
	return this.error;
};

Neuron.prototype.setWeights = function(weights){
	this.weights = weights;
};

Neuron.prototype.getWeights = function(){
	return this.weights;
};

Neuron.prototype.setWeight = function(index, value){
	this.weights[index] = value;
};

Neuron.prototype.getWeight = function(index){
	return this.weights[index];
};

Neuron.prototype.setPrevDeltas = function(prevDeltas){
	this.prevDeltas = prevDeltas;
};

Neuron.prototype.getPrevDeltas = function(){
	return this.prevDeltas;
};

Neuron.prototype.setPrevDelta = function(index, value){
	this.prevDeltas[index] = value;
};

Neuron.prototype.getPrevDelta = function(index){
	return this.prevDeltas[index];
};

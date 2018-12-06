/*
**	@data 						-> Multidimensional array inputs and outputs
**	@hiddenNeurons 				-> Hidden neuron number
**	@outputNeurons 				-> Output neuron number
**	@error 						-> Training process will be stop if the error lower than this number
**	@epoch 						-> Number of iterations
**	@learningRate 				-> Learning rate
**  @useAdaptiveLearningRate 	-> If you don't use adaptive learning rate mechanism you can set this value as false (default => true)
**  If you don't set momentum and weight decay values network automatically ignore these processes
*/
function Network(data, hiddenNeurons, outputNeurons, error, epoch, learningRate, useAdaptiveLearningRate=true){
	this.trainingData = data;
	this.inputNeurons = data[0].length - outputNeurons;
	this.hiddenNeurons = hiddenNeurons;
	this.outputNeurons = outputNeurons;
	this.error = error;
	this.epoch = epoch;
	this.learningRate = learningRate;
	this.testData;
	this.trainingError = null;
	this.testError = null;
	this.weightDecay = 0;
	this.momentum = 0;
	this.adaptiveLearningRateA = 0.001;
	this.adaptiveLearningRateB = 0.001;
	this.useAdaptiveLearningRate = useAdaptiveLearningRate;
}

Network.prototype.init = function(){
	var inputNeuronCount = this.inputNeurons;
	var hiddenNeuronCount = this.hiddenNeurons;
	var outputNeuronCount = this.outputNeurons;

	this.inputNeurons = [];
	this.hiddenNeurons = [];
	this.outputNeurons = [];

	for(var i = 0; i < inputNeuronCount; i++){
		var neuron = new Neuron();
		neuron.initNeuron(hiddenNeuronCount);
		this.inputNeurons.push(neuron);
	}

	var biasNeuron = new Neuron();
	biasNeuron.initNeuron(hiddenNeuronCount);
	biasNeuron.setOutput(1);
	this.inputNeurons.push(biasNeuron);

	for(var i = 0; i < hiddenNeuronCount; i++){
		var neuron = new Neuron();
		neuron.initNeuron(outputNeuronCount);
		this.hiddenNeurons.push(neuron);
	}

	biasNeuron = new Neuron();
	biasNeuron.initNeuron(outputNeuronCount);
	biasNeuron.setOutput(1);
	this.hiddenNeurons.push(biasNeuron);

	for(var i = 0; i < outputNeuronCount; i++){
		var neuron = new Neuron();
		this.outputNeurons.push(neuron);
	}
};

Network.prototype.train = function(){
	var prevError = 0;
	for(var i = 0; i < this.epoch; i++){

		this.shuffleArray(this.trainingData);
		var totalError = 0;
		for(var j = 0; j < this.trainingData.length; j++){

			this.feedForward(this.trainingData[j]);
			totalError += this.calculateError();
			this.backPropagation();

		}
		totalError = totalError / this.trainingData.length;
		this.trainingError = totalError;

		/*
		*	-> Adaptive Learning Rate Start
		*/
		if(this.useAdaptiveLearningRate){
			if(prevError < totalError){
				this.learningRate -= this.learningRate * this.adaptiveLearningRateB;
			}else{
				this.learningRate += this.adaptiveLearningRateA;
			}
		}
		prevError = totalError;
		/*
		*	-> Adaptive Learning Rate End
		*/
		// We add weight sum to error for prevent the weight increasing so much
		totalError += (this.weightDecay/2) * this.getSumOfWeightSquares();
		if(totalError < this.error)
			break;
	}
};

Network.prototype.test = function(){
	var totalError = 0;
	for(var i = 0; i < this.testData.length; i++){

		this.feedForward(this.testData[i]);
		totalError += this.calculateError();

	}
	this.testError = totalError / this.testData.length;
};

Network.prototype.backPropagation = function(){
	for(var i = 0; i < this.outputNeurons.length; i++){

		var outputError = this.outputNeurons[i].getError();

		for(var j = 0; j < this.hiddenNeurons.length; j++){
			var accumulateOutputError = 0;
			if(i != 0){
				accumulateOutputError = this.hiddenNeurons[j].getError();
			}
			var weight = this.hiddenNeurons[j].getWeight(i);
			this.hiddenNeurons[j].setError(accumulateOutputError + (outputError * weight));

			var delta = this.learningRate * outputError * this.hiddenNeurons[j].getOutput();
			var prevDelta = this.hiddenNeurons[j].getPrevDelta(i);
			var newWeight = weight + delta;
			if(this.useWeightDecay) newWeight -= this.weightDecay * weight;
			if(this.useMomentum) newWeight += this.momentum * prevDelta;
			this.hiddenNeurons[j].setWeight(i, newWeight);
			this.hiddenNeurons[j].setPrevDelta(i, delta);
		}
	}

	for(var i = 0; i < this.hiddenNeurons.length-1; i++){ // Because of bias

		var accumulatedError = this.hiddenNeurons[i].getError();
		var derivative = this.hiddenNeurons[i].getOutput() * (1 - this.hiddenNeurons[i].getOutput());

		for(var j = 0; j < this.inputNeurons.length; j++){
			var delta = this.learningRate * accumulatedError * derivative * this.inputNeurons[j].getOutput();

			var weight = this.inputNeurons[j].getWeight(i);
			var prevDelta = this.inputNeurons[j].getPrevDelta(i);

			var newWeight = weight + delta;
			if(this.useWeightDecay) newWeight -= this.weightDecay * weight;
			if(this.useMomentum) newWeight += this.momentum * prevDelta;

			this.inputNeurons[j].setWeight(i, newWeight);
			this.inputNeurons[j].setPrevDelta(i, delta);
		}
	}
};

Network.prototype.calculateError = function(){
	var totalError = 0;
	for(var i = 0; i < this.outputNeurons.length; i++){
		var error = this.outputNeurons[i].getActual() - this.outputNeurons[i].getOutput();
		this.outputNeurons[i].setError(error);
		totalError += Math.pow(error, 2);
	}
	return totalError;
};

Network.prototype.feedForward = function(array){
	this.setOutputs(array);

	for(var i = 0; i < this.hiddenNeurons.length-1; i++){ // Because of bias
		var input = 0;
		for(var j = 0; j < this.inputNeurons.length; j++){
			input += this.inputNeurons[j].getOutput() * this.inputNeurons[j].getWeight(i);
		}
		this.hiddenNeurons[i].setOutput(this.sigmoid(input));
	}

	for(var i = 0; i < this.outputNeurons.length; i++){
		var input = 0;
		for(var j = 0; j < this.hiddenNeurons.length; j++){
			input += this.hiddenNeurons[j].getOutput() * this.hiddenNeurons[j].getWeight(i);
		}
		this.outputNeurons[i].setOutput(input);
	}
};

Network.prototype.getSumOfWeightSquares = function(){
	var sum = 0;
	for(var i = 0; i < this.outputNeurons.length; i++){
		for(var j = 0; j < this.hiddenNeurons.length; j++){
			sum += Math.pow(this.hiddenNeurons[j].getWeight(i), 2);
		}
	}
	for(var i = 0; i < this.hiddenNeurons.length - 1; i++){
		for(var j = 0; j < this.inputNeurons.length; j++){
			sum += Math.pow(this.inputNeurons[j].getWeight(i), 2);
		}
	}
	return sum;
};

Network.prototype.getRSquare = function(){
	var actuals = [];
	var outputs = [];
	for(var i = 0; i < this.testData.length; i++){
		this.feedForward(this.testData[i]);
		actuals.push(this.outputNeurons[0].getActual());
		outputs.push(this.outputNeurons[0].getOutput());
	}
	var meanActuals = 0;
	for(var i = 0; i < actuals.length; i++){
		meanActuals += actuals[i];
	}
	meanActuals = meanActuals / actuals.length;
	var sumOfSquares = 0;
	var sumOfSquaresOfResiduals = 0;
	for(var i = 0; i < actuals.length; i++){
		sumOfSquares += Math.pow(actuals[i] - meanActuals, 2);
		sumOfSquaresOfResiduals += Math.pow(actuals[i] - outputs[i], 2);
	}
	var rSquare = 1 - sumOfSquaresOfResiduals / sumOfSquares;
	return [rSquare, actuals, outputs];
};

Network.prototype.setWeightDecay = function(weightDecay){
	this.weightDecay = weightDecay;
};

Network.prototype.setMomentum = function(momentum){
	this.momentum = momentum;
};

Network.prototype.setTestData = function(testData){
	this.testData = testData;
};

Network.prototype.sigmoid = function(value){
	return 1 / (1 + Math.exp(-1 * value));
};

Network.prototype.setOutputs = function(array){
	var i = 0;
	for(; i < this.inputNeurons.length-1; i++){ // Because of bias
		this.inputNeurons[i].setOutput(array[i]);
	}
	for(var j = 0; j < this.outputNeurons.length; j++){
		this.outputNeurons[j].setActual(array[i]);
		i++;
	}
};

Network.prototype.shuffleArray = function(array) {
	for (var i = array.length - 1; i > 0; i--) {
		var j = Math.floor(Math.random() * (i + 1));
		var temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
};

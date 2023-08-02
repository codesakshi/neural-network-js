/**
* Neural Network implementation
*
* Can add Multiple layers
*
* Neuron in each layer is considered together.
* This will help reduce code for forward and back workd propagation
*
* Support bias for each neuron
* bias will be the last entry in weights[][]
*/

// NNS namespace
var NNS = (function() {
	
	// Activation Function
	function ActivationFunction(name, output, derivative) {
		
		if (typeof(output) != 'function' || typeof(derivative) != 'function') {
		  throw 'ActivationFunction requires two functions';
		}
		
		this.name = name;
		this.output = output;
		this.derivative = derivative;
	};

	// sigmoid activation function and derivative
	var sigmoid = new ActivationFunction( 
		"sigmoid",
		x => 1 / (1 + Math.exp(-x)), // Activation
		x => x * (1 - x) // Derivative
	);

	// tanh activation function and derivative
	var tanh = new ActivationFunction(
		"tanh",
		x => Math.tanh(x), // Activation
		x => (1 - Math.pow(Math.tanh(x), 2)) // Derivative
	);

	// leakyRelu activation function and derivative
	var leakyRelu = new ActivationFunction(
		"leakyRelu",
		x => ( x > 0 ? x : x*0.01 ), // Activation
		x => ( x > 0 ? 1 : 0.01 ) // Derivative
	);

	// binaryStep activation function and derivative
	var binaryStep = new ActivationFunction(
		"binaryStep",
		x => ( x > 0 ? 1 : 0 ), // Activation
		x => ( 1 ) // Derivative ( It is actually 0. To get valid output; we are using 1)
	);

	// Map to get activationFunction for a given name.
	var activationFunctions = {
		[sigmoid.name] : sigmoid,
		[tanh.name] : tanh,
		[leakyRelu.name] : leakyRelu,
		[binaryStep.name] : binaryStep,
	}

	/**
	* NeuralNetwork class
	*/
	var NeuralNetwork = function ( inputCount, name){

		// Number of inputs
		this.inputCount = inputCount;
		
		// identifier
		this._name = name; // '_' is added. So it will be added as first entry in the json
		
		// Stores Neural layers
		this.layers = [];
		
		// Add a layer.
		// Arguments 
		//		neuronCount - number of neuron in this layer
		//		activation - Activation Function
		this.addLayer = function( neuronCount, activation ){
			
			if (neuronCount === undefined || activation === undefined) {
				throw 'addLayer requires valid arguments';
			}
			
			if( !(activation instanceof ActivationFunction) ) {
				throw 'activation should be instance of ActivationFunction!';
			}
			
			if ( neuronCount <= 0) {
				throw 'Layer must have a positive number of neurons!';
			}
			
			let layer = {
				neurons : Array(neuronCount),// two diamentional array to store all neuron of this layer
				activation : activation, // activation function
				outputs : [], // one dimentional array to store all outputs of this layer.
				deltas : [], // one dimentional array to store deltas from next layer.
			 }
			
			this.layers.push( layer );
		 
			return this;
		}
		
		// Fill the neuron weights with random numbers
		// additial weight is added to each neuron for storing 'bias'
		this.randomizeWeights = function(){
			
			for( let i = 0; i < this.layers.length; i ++ ){
				
				let layer = this.layers[i];
				
				// For first layer, weightsCount is number of inputs
				// Other layers, weightsCount is number of neurons in previous layer.
				let weightsCount = i == 0 ? 
					this.inputCount : this.layers[i-1].neurons.length;
				
				// incriment count for adding bias weight
				weightsCount ++;
				
				let neurons = layer.neurons;
				
				let neuronCount = neurons.length;
				
				for( let j = 0; j < neuronCount; j ++ ){
					
					let weights = [];
					
					for (let k = 0; k < weightsCount; k ++) {
						weights[k] = Math.random() * 2 - 1;
					}
					
					neurons[j] = weights;
				}
			}
			
			return this;
		}
		
		// predict the output for given inputs
		this.predict = function( inputs ){
			this.forward( inputs );
			return this.getOutput();
		}
		
		// Do forward propagation
		this.forward = function(inputs ) {

			for( let i = 0; i < this.layers.length; i ++) {

				// If it is first layer, input is user input
				// for other layers, input will be output from previous layer.
				let inputToLayer = (i == 0 )? 
					inputs : this.layers[i-1].outputs;
					
				let layer = this.layers[i];
				let neuronCount = layer.neurons.length;
				
				for (let j = 0; j < neuronCount; j ++) {

					let weights = layer.neurons[j];
					
					// Initialize value with bias weight 
					// index is actually [weights.length-1]
					// 'inputToLayer.length' is used to throw error if any mismatch in size.
					let value = weights[inputToLayer.length];
					// Do computation of output
					for (let k = 0; k < inputToLayer.length; k ++) {
						value += inputToLayer[k] * weights[k];
					}
					
					// store output in outputs array
					layer.outputs[j] = layer.activation.output(value);
				}
			}
		}
		
		// Return output of the NeuralNetwork
		// Output will be outputs of the final layer.
		this.getOutput = function() {
			let outputs = this.layers[this.layers.length-1].outputs;
			return outputs.length == 1 ? outputs[0] : outputs;
		}
		
		// Train the NeuralNetwork
		this.train = function( inputs, actuals, learningRate = 0.01 ) {

			this.forward( inputs);

			this.backward( inputs, actuals );
			
			this.updateWeights( inputs, learningRate );
		}
		
		this.backward = function( inputs, actuals ) {
			
			// iterate from end to start
			for( let i = this.layers.length - 1; i >=0 ;i-- ){
				
				let layer = this.layers[i];
				let neuronCount = layer.neurons.length;
				
				let outputs = layer.outputs;
				
				if( i == ( this.layers.length - 1 ) ){
					// for last layer, error depends on output and actual
					for( let j =0; j < neuronCount; j ++ ){
						let error = outputs[j] - actuals[j];
						// Set delta
						layer.deltas[j] 
							= error * layer.activation.derivative( outputs[j] );
					}
					
				} else {
					
					// Error is cumulative error from next layer.
					let nextLayer = this.layers[i+1];
					let nextLayerNeuronCount = nextLayer.neurons.length;
					for( let j =0; j < neuronCount; j ++ ){
						
						let error = 0.0;
						for (let k = 0; k < nextLayerNeuronCount; k ++) {
							error += nextLayer.neurons[k][j] * nextLayer.deltas[k];
						}
						// Set delta
						layer.deltas[j] 
							= error * layer.activation.derivative( outputs[j] );
					}
				}
			}
		}
		
		this.updateWeights = function( inputs, learningRate){
			
			for( let i = 0; i < this.layers.length; i ++) {
				
				let inputToLayer = (i == 0 ) ? 
					inputs : this.layers[i-1].outputs;
					
				let layer = this.layers[i];
				
				let neuronCount = layer.neurons.length;
				
				for (let j = 0; j < neuronCount; j ++) {
					
					let weights = layer.neurons[j];
					for (let k = 0; k < inputToLayer.length; k ++) {
						
						weights[k] -= learningRate * layer.deltas[j] * inputToLayer[k];
					}
					
					// Update bias weight
					weights[inputToLayer.length] -= learningRate * layer.deltas[j];
				}
			}
		}
		
		this.toJsonString = function(){
			let fields = [
				"_name",
				"inputCount",
				"layers",
				"neurons",
				"activation",
				"name"
			];
			return JSON.stringify(this, fields, 4);
		}
		
		this.load = function( json ){
		
			if (json.inputCount === undefined ){
				throw 'inputCount is undefined';
			}
			
			// Number of inputs
			this.inputCount = json.inputCount;
			
			// identifier
			this._name = json._name; 
						
			for ( let layer of json.layers) {
				// update activation name with activation function
				layer.activation = activationFunctions[layer.activation.name];
				layer.outputs = [];
				layer.deltas = [];
				
				// Add the layer
				this.layers.push( layer );
			}
			
		}
	};
	
	return {
		ActivationFunction: ActivationFunction,
		NeuralNetwork: NeuralNetwork,
		sigmoid : sigmoid,
		tanh : tanh,
		leakyRelu : leakyRelu,
		binaryStep : binaryStep
  };

})();

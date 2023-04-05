/**
* AI Neural Network implementaion
* 
* Can add multiple layers
*
* Nodes ( neuron ) in each layer is considered together.
* This will help reduce code for forward and backword propagation
*
* Support bias for each node ( neuron )
* bias will be the last entry in wieghts[][]
*
**/

// Actiation function
function ActivationFunction( name, output, derivative ){
	
	if( typeof(output) != 'function' || typeof(output) != 'function' ){
		throw 'ActivationFunction requires two functions';
	}

	this.name = name;
	this.output = output;
	this.derivative = derivative;
}

// Sigmoid activation function and derivative
var sigmoid = new ActivationFunction(
	"sigmoid",
	x => 1/( 1 + Math.exp(-x)), // Activation
	x => x * ( 1-x ) // Derivative
);


// Tanh activation function and derivative
var tanh = new ActivationFunction(
	"tanh",
	x => Math.tanh(x), // Activation
	x => ( 1 - Math.pow( Math.tanh(x) ),2 ) // Derivative
);

// LeakyRelu activation function and derivative
var leakyRelu = new ActivationFunction(
	"leakyRelu",
	x => (x > 0 ? x : x * 0.01), // Activation
	x => (x > 0 ? 1 : 0.01) // Derivative
);

// Binary step activation function and derivative
var binaryStep = new ActivationFunction(
	"binaryStep",
	x => (x > 0 ? 1 : 0), // Activation
	x => ( 1 ) // Derivative ( Actually binary step derivative is 0 )
);

var activationFunctions = {
	"sigmoid" : sigmoid,
	"tanh" : tanh,
	"leakyRelu" : leakyRelu,
	"binaryStep" : binaryStep,
};

/**
* Neural network class
**/
function NeuralNetwork( inputCount, name ){

	// Number of inputs
	this.inputCount = inputCount;

	// identifier
	this._name = name; // '_' is there, so it will appear at top of the generated json

	// Stores Neural layers
	this.layers = [];

	// Add a layer
	// Arguments
	//	nodeCount - number of nodes ( neurons ) in the layer
	//	activation - ActivationFunction for the layer
	this.addLayer = function( nodeCount, activation ){

		if( nodeCount === undefined || activation === undefined ){
			throw 'addlayer requires valid inpputs';
		}

		if( ! (activation instanceof ActivationFunction) ){
			throw 'activation should be instance of ActivationFunction';
		}

		if( nodeCount <=0 ){
			throw 'Layer must have a positive number of nodes';
		}

		let layer = {
			nodeCount : nodeCount,
			activation : activation,
			weights : [],
			outputs : [],
			deltas : [],
		};

		this.layers.push( layer );

		return this;
	}

	// Fill the weights with random numbers
	// Additional weight is added at the end of each node for storing weights
	this.randomizeWeights = function(){

		for( let i = 0; i < this.layers.length; i ++ ){

			let layer = this.layers[i];

			// For first layer, weightsCount is number of inputs
			// Other layers, weightsCount is number of nodes in previous layer.
			let weightsCount = (i == 0) ?
				this.inputCount : this.layers[i-1].nodeCount;

			//incriment count for storing 'bias' weight
			weightsCount ++;

			let weights = layer.weights;

			for( let j = 0; j < layer.nodeCount; j ++ ){

				weights[j] = [];

				for( let k = 0; k < weightsCount; k ++ ){
					weights[j][k] = Math.random() * 2 - 1;
				}
			}
		}

		return this;
	}

	// predict the output for given inputs
	this.predict = function( inputs ){
		this.forward( inputs );
		return this.getOuput();
	}

	// Do forward propagation
	this.forward = function( inputs ){

		for( let i = 0; i < this.layers.length; i ++ ){

			let layer = this.layers[i];

			// If it is first layer, input is user input
			// for other layers, input will be output of previous layer.

			let inputToLayer = ( i ==0 ) ?
				inputs : this.layers[i-1].outputs;

			for( let j = 0; j < layer.nodeCount; j ++ ){

				let weights = layer.weights[j];

				// Initialize the value with bias weight;
				// 'index' is actually 'weights.length-1'
				// 'inputToLayer.length' is used to throw error if there is any mismatch in size.
				let value = weights[inputToLayer.length];
				for( let k = 0; k < inputToLayer.length; k ++ ){
					value += inputToLayer[k] * weights[k];
				}

				// store the outputs in outputs array
				layer.outputs[j] = layer.activation.output( value );
			}
		}
	}
	
	
	// Return output of the NeuralNetwork
	// Output will be outputs of the final layer.
	this.getOuput = function() {
		let outputs = this.layers[this.layers.length - 1].outputs;
		return outputs.length == 1 ? outputs[0] : outputs;
	}
	
	this.train = function ( inputs, actuals, learningRate = 0.01 ) {
		
		this.forward( inputs );
		
		this.backword( inputs, actuals );
		
		this.updateWeights( inputs, learningRate );
	}
		
	this.backword= function ( inputs, actuals ){
		
		// iterate from end to start
		for( let i = this.layers.length - 1; i >=0 ; i -- ){
			
			let layer = this.layers[i];
			
			let outputs = layer.outputs;
			
			if( i == ( this.layers.length - 1 )){
				
				// for last layer, error depends on outputs and actuals
				for( let j = 0; j < layer.nodeCount; j ++ ){
					let error = outputs[j] - actuals[j];
					
					// set deltas
					layer.deltas[j] = 
						error * layer.activation.derivative( outputs[j] );
				}
			} else{
				
				// Error is cumulative error from next layer.
				let nextLayer = this.layers[i+1];
				
				for( let j = 0; j < layer.nodeCount; j ++ ){
					
					let error = 0.0;
					for( let k = 0; k < nextLayer.nodeCount; k ++ ){
						error += nextLayer.weights[k][j] * nextLayer.deltas[k];
					}
					// Set delta
					layer.delta[j] = 
						error * layer.activation.derivative( outputs[j] );
				}
			}
		}
	}

	this.updateWeights =  function( inputs, learningRate ){
		
		for( let i = 0; i < this.layers.length; i ++ ){

			let layer = this.layers[i];

			// If it is first layer, input is user input
			// for other layers, input will be output of previous layer.

			let inputToLayer = ( i ==0 ) ?
				inputs : this.layers[i-1].outputs;

			for( let j = 0; j < layer.nodeCount; j ++ ){

				let weights = layer.weights[j];
				
				for( let k = 0; k < inputToLayer.length; k ++ ){
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
			"nodeCount",
			"activation",
			"name",
			"weights",
		];
		
		return JSON.stringify( this, fields, 4);
	}
	
}

/** Load NeuralNetwork from json **/
function networkFromJson( json ){
	
	if( json.inputCount === undefined ){
		throw 'inputCount is undefined';
	}
	
	let network = new NeuralNetwork( json.inputCount, json._name );
	
	for( let layer of json.layers ) {
		
		// update activation name with activation function
		layer.activation = activationFunctions[layer.activation.name];
		layer.outputs = [];
		layer.deltas = [];
		
		network.layers.push(layer);
	}
	
	return network;
}


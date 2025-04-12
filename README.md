I have been messing around with RNNs recently; there are many kinds - 
though LSTMs, GRUs, and vanilla RNNs are most commonly encountered.

Here is the equation for a vanilla RNN, using Einstein summation notation:


$$h^{(t)}_{i} = f\Bigl( A_{ij}\, h^{(t-1)}_{j} + B_{ij}\, x^{(t)}_{j} + c_{i} \Bigr)$$


In this equation:
- $$h^{(t)}_{i}$$ is the hidden state at time t,
- $$A_{ij}$$ is the weight matrix applied to the previous hidden state,
- $$B_{ij}$$ is the weight matrix applied to the input at time t,
- $$c_{i}$$ is the bias vector,
- f is some activation function.

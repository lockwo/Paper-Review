import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow import StateFn, Z, I, CircuitSampler, Gradient
from qiskit.algorithms.optimizers import GradientDescent
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeBogota
from qiskit.providers.aer.noise import NoiseModel

def main(backend, user_message, shots, init_point):
    def paper_circuit():
        circuit = QuantumCircuit(5)
        params = ParameterVector("theta", length=5)
        for i in range(5):
            circuit.rx(params[i], i)
        circuit.cx(0, 1)
        circuit.cx(2, 1)
        circuit.cx(3, 1)
        circuit.cx(4, 3)
        hamiltonian = I
        for i in range(1, 5):
            if i == 3:
                hamiltonian = hamiltonian ^ Z
            else:
                hamiltonian = hamiltonian ^ I
        readout_operator = StateFn(hamiltonian, is_measurement=True) @ StateFn(circuit)
        return circuit, readout_operator

    def evaluate_expectation(x):
        value_dict = dict(zip(circuit.parameters, x))
        result = sampler.convert(op, params=value_dict).eval()  
        return np.real(result)

    def evaluate_gradient(x):
        value_dict = dict(zip(circuit.parameters, x))
        result = sampler.convert(gradient, params=value_dict).eval()
        return np.real(result)

    def gd_callback(nfevs, x, fx, stepsize):
        if nfevs % 10 == 0:
            user_message.publish([nfevs, fx])
        gd_loss.append(fx)
        x_values.append([x[0], x[1]])
        
        
    circuit, op = paper_circuit()
    
    device_backend = FakeBogota()
    backend = Aer.get_backend('aer_simulator')
    device = QasmSimulator.from_backend(device_backend)
    coupling_map = device.configuration().coupling_map
    noise_model = NoiseModel.from_backend(device)
    q_instance = QuantumInstance(backend=backend, coupling_map=coupling_map, noise_model=noise_model, shots=shots)
    
    #q_instance = QuantumInstance(backend, shots=shots)
    sampler = CircuitSampler(q_instance) 
    gradient = Gradient(grad_method='param_shift').convert(op)
    gd_loss, x_values = [], []
    gd = GradientDescent(maxiter=100, learning_rate=0.1, tol=1e-4, callback=gd_callback)
    gd.optimize(init_point.size, evaluate_expectation, gradient_function=evaluate_gradient, \
                initial_point=init_point) 
    x_values.clear()
    gd_values = gd_loss.copy()
    gd.optimize(init_point.size, evaluate_expectation, gradient_function=evaluate_gradient, \
                initial_point=[0.1, 0.15, 0, 0, 0]) 
    x_values = np.array(x_values)
    return gd_values, x_values


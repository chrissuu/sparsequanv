import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import numpy as np
import time
import numpy as np
import torch.nn as nn
from utils import *

class SparseQuanvLayer(nn.Module):
    
    def __init__(self, rand_params, WIRES, dev, bz, input_shape, PRINT, kernel_shape = (2,2,2)):
        super().__init__()
        self.rand_params = rand_params
        self.WIRES = WIRES
        self.dev = dev
        self.bz = bz
        self.input_shape = input_shape
        (self.w, self.l, self.h) = input_shape
        self.kernel_shape = kernel_shape
        (self.kw, self.kl, self.kh) = kernel_shape
        self.print = PRINT

        if self.w % self.kw == 1:
            self.w -= 1

        if self.l % self.kl == 1:
            self.l -= 1
        
        if self.h % self.kh == 1:
            # print("HERE")
            self.h -= 1
            

        
    def circuit(self, phi):
        # Encoding of KERNEL classical input values
        for j in range(self.WIRES):
            qml.RY(np.pi * phi[j], wires=j)
        # print("qml.RY(np.pi * phi[j], wires = j") 
        # Random quantum circuit
        RandomLayers(self.rand_params, wires=list(range(self.WIRES)))
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(self.WIRES)]

    # produces WIRES kerneled images of size 14,14 per image
    def quanv_sparse(self, image):
        tot = np.zeros((self.bz, self.w // self.kw, self.l // self.kl,\
                        self.h // self.kh, self.WIRES))
        
        # Loop over the coordinates of the top-left pixel of dim kernel_dim^3 squares
        start = time.time()
        for img in range(0, self.bz):
            for j in range(0, self.l, self.kl):
                for k in range(0, self.w, self.kw):
                    for m in range(0, self.h, self.kh):
                        # Process a squared 2x2 region of the image with a quantum circuit
                        # q_results = self.circuit(
                        #     [
                        #         image[j, k, m],
                        #         image[j + 1, k + 1, m],
                        #         image[j, k + 1, m+1],
                        #         image[j + 1, k, m+1],
                        #     ]
                        # )
                        qnode = qml.QNode(self.circuit,self.dev)
                        (im_b, im_l, im_w, im_h) = image.shape
                        assert(img < im_b)
                        assert(j < im_l)
                        assert(k < im_w)
                        assert(m < im_h)
                        # print(image.shape)
                        q_results = qnode(
                            [
                                image[img, j, k, m],
                                image[img, j + 1, k + 1, m],
                                image[img, j, k + 1, m+1],
                                image[img, j + 1, k, m+1],
                            ]
                        )

                        
                        # Assign expectation values to different channels of the output pixel (j/2, k/2)
                        for c in range(self.WIRES):
                            # print(q_results[0])
                            tot[img, j // self.kl, k // self.kw, m // self.kh, c] = q_results[c]
                            # print(self.WIRES)

        end = time.time()

        if self.print:

            print(f"Image processing for sparse quanv took {end-start} seconds")
        return tot
    

    def forward(self, x):
        return self.quanv_sparse(x)

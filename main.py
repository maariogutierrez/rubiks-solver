from rubiks_engine import RubiksEngine
from nn import train, CubeValueNet
import torch

if __name__ == '__main__':
    cube = RubiksEngine()
    net = CubeValueNet()
    train(net, cube, epochs=10000)
    torch.save(net.state_dict(), 'rubiks_model.pth')
    print("Model weights saved to rubiks_model.pth")
# -*- encoding: utf-8 -*-
import torch

# 使用torch.add()张量相加
# 使用torch.sub()张量相减
# 使用torch.mul()张量相乘
# 使用torch.div()张量相除
a = torch.randn(4)
print(a)
print(torch.add(a, 20))
b = torch.randn(4)
print(b)
print(torch.add(a, b))
'''
tensor([ 1.5413, -1.0359,  0.1513,  0.7180])
tensor([21.5413, 18.9641, 20.1513, 20.7180])
tensor([-0.3158, -1.6348, -1.6302,  0.0972])
tensor([ 1.2255, -2.6707, -1.4789,  0.8152])
'''

'''
1)torch.addcdiv(input, value=1, tensor1, tensor2, out=None)
  addcmul(input, value=1, tensor1, tensor2, out=None)
  用tensor2对tensor1逐元素相除，然后乘以标量值value 并加到input
  公式表达为：input+value*tensor1/tensor2
  公式表达为：input+value*tensor1*tensor2
'''
# 使用torch.addcdiv(张量相加和相除)
t = torch.randn(1, 3)
t1 = torch.randn(3, 1)
t2 = torch.randn(1, 3)
print(torch.addcdiv(t, 0.1, t1, t2))

'''
torch.log(input,out=None)#计算input的自然对数
torch.log10(input,out=None)#计算input的10为底的对数
torch.log2(input,out=None)#计算input的2为底的对数
torch.exp(input,out=None)#对输入input按元素求e次幂值，并返回结果张量，幂值e可以为标量也可以是和input相同大小的张量
torch.pow(input,out=None)#次方运算

torch.abs(input,out=None)#计算张量的每个元素绝对值
torch.acos(input,out=None)#返回一个新张量，包含输入张量每个元素的反余弦
torch.cosh(input,out=None)#返回一个新张量，包含输入input张量每个元素的双曲余弦
torch.cos(input,out=None)#返回一个新张量，包含输入input张量每个元素的余弦
torch.asin(input,out=None)#返回一个新张量，包含输入input张量每个元素的反正弦
torch.atan(input,out=None)#返回一个新张量，包含输入input张量每个元素的反正切
torch.atan2(input1, input2, out=None)#返回一个新张量，包含两个输入张量input1和input2的反正切函数
'''
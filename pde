Exp 1 : Vector and Linear Combination
#1
import numpy as np
import sympy as sp
def isp(vec,s):
    n=len(s)
    var=sp.symbols(f'a:{n}')
    m=-vec
    for i,v in enumerate(var):
        m=m+v*s[i]
    sca=sp.solve(m,var)
    if len(sca)==0:
        print("No! The given vector is not in the given span of S")
        return False
    else:
        print("The given vector is in the given span of S")
    return sca

vec=np.array((2,-1,1))
s=np.array([(1,0,2),(-1,1,1)])
isp(vec,s)

#2
import numpy as np
import sympy as sp
def isp(vec,s):
    n=len(s)
    var=sp.symbols(f'a:{n}')
    m=-vec
    for i,v in enumerate(var):
        m=m+v*s[i]
    sca=sp.solve(m,var)
    if len(sca)==0:
        print("No! The given vector is not in the given span of S")
        return False
    else:
        print("The given vector is in the given span of S")
    return sca
vec=sp.Matrix([[1,2],[-3,4]])
v1=sp.Matrix([[1,0],[-1,0]])
v2=sp.Matrix([[0,1],[0,1]])
v3=sp.Matrix([[1,1],[0,0]])
s=np.array([v1,v2,v3])
isp(vec,s)

Exp 2 : Linear (in)dependence, Basis, and Dimension
#1
import numpy as np
import sympy as sp
from sympy.abc import *
def is_independent(S):
    alphas = sp.symbols(f'a:{len(S)}')
    M = 0
    for i,var in enumerate(alphas):
        if M is None:
            M = var*S[i]
        else:
            M = M + var*S[i]
    soln = sp.solve(M, alphas, manual=True)
    if np.abs(soln[0]).sum()==0:
        print("The given set is linearly independent")
        return True
    else:
        print("The given set is linearly dependent")
        return False

def is_basis(B,dim):
    if is_independent(B):
        if len(B) == int(dim):
            print('And the dimension of V = {dim} = number of elements of B.')
            print('The given set is a Basis')
            return True
        else:
            print("The dimension is not matched. So it's not a basis")
            return False
    else:
        print("The given set is not a basis")
        return False

B = np.array([[1,0,0,-1],[0,1,0,-1],[0,0,1,-1],[0,0,0,1]])
dim = 4
is_basis(B,dim)

#2
import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from sympy.abc import *
def is_independent(S):
    alphas = sp.symbols(f'a:{len(S)}')
    M = 0
    for i,var in enumerate(alphas):
        if M is None:
            M = var*S[i]
        else:
            M = M + var*S[i]
    soln = sp.solve(M, alphas, manual=True)
    if np.abs(soln[0]).sum()==0:
        print("The given set is linearly independent")
        return True
    else:
        print("The given set is linearly dependent")
        return False

def is_basis(B,dim):
    if is_independent(B):
        if len(B) == int(dim):
            print('And the dimension of V = {dim} = number of elements of B.')
            print('The given set is a Basis')
            return True
        else:
            print("The dimension is not matched. So it's not a basis")
            return False
    else:
        print("The given set is not a basis")
        return False

B = np.array([[1+x+x**2],[x+x**2],[x**2]])
dim = 4
is_basis(B,dim)








Exp 3 : Linear Transformations
#1
import sympy as sp
import numpy as np

def T(v):
    return np.array([v[0], 2*v[0] + v[1]])

def is_linear(T, domain_dim):
    x = np.array(sp.symbols(f'x:{domain_dim}'))
    y = np.array(sp.symbols(f'y:{domain_dim}'))  
    c = sp.symbols('c')
    lhs = T(c*x + y)
    rhs = c*T(x) + T(y)
    lhs = [sp.expand(i) for i in lhs]
    print(f'T(c*x+y)={lhs}')
    rhs = [sp.expand(i) for i in rhs]
    print(f'c*T(x)+T(y)={rhs}')
   
    if lhs == rhs:
        print("The given map is a linear transformation")
        return True
    else:
        print("The given map is not a linear transformation")
        return False

is_linear(T, 2)

#2
import sympy as sp
import numpy as np

def T(v):
    return np.array([v[0] - v[1], 2*v[2],1])

def is_linear(T, domain_dim):
    x = np.array(sp.symbols(f'x:{domain_dim}'))
    y = np.array(sp.symbols(f'y:{domain_dim}'))
    c = sp.symbols('c')
    lhs = T(c*x + y)
    rhs = c*T(x) + T(y)
    lhs = [sp.expand(i) for i in lhs]
    print(f'T(c*x+y)={lhs}')
    rhs = [sp.expand(i) for i in rhs]
    print(f'c*T(x)+T(y)={rhs}')  
   
    if lhs == rhs:
        print("The given map is a linear transformation")
        return True
    else:
        print("The given map is not a linear transformation")
        return False

is_linear(T, 3)

#3
import sympy as sp
import numpy as np

def T(v):
    return np.array([v[1] + v[2], v[0] + v[2], v[0] + v[1]])

def is_linear(T, domain_dim):
    x = np.array(sp.symbols(f'x:{domain_dim}'))
    y = np.array(sp.symbols(f'y:{domain_dim}'))
    c = sp.symbols('c')
    lhs = T(c*x + y)
    rhs = c*T(x) + T(y)
    lhs = [sp.expand(i) for i in lhs]
    print(f'T(c*x+y)={lhs}')
    rhs = [sp.expand(i) for i in rhs]
    print(f'c*T(x)+T(y)={rhs}')

    if lhs == rhs:
        print("The given map is a linear transformation")
        return True
    else:
        print("The given map is not a linear transformation")
        return False

is_linear(T, 3)





Exp 4 : Finding matrix of linear transformations
#1
from numpy import *
from numpy import linalg as la
def T(x,y,z):
    return [x-y+z , 2*x+3*y-(1/2)*z , x+y-2*z]
B1 = array([[-1,1,0],[5,-1,2],[1,2,1]])
print("The specified basis for the domain space is \n",B1)
B2 = array([[1,1,0],[0,0,1],[1,5,2]])
print("The specified basis for the co-domain space is \n",B2)

w1 = T(-1,1,0)
w2 = T(5,-1,2)
w3 = T(1,2,1)
print("Evaluating T at the vectors of basis B1: \n ",w1,w2,w3)

v1 = la.solve(B2.T,w1)
v2 = la.solve(B2.T,w2)
v3 = la.solve(B2.T,w3)

M = array([v1,v2,v3])

print("The matrix of the linear transofrmation is :\n",M)

#2
import sympy as sp
import numpy as np
def lt_of_matrix(B1,B2,M):
    k,m=len(B1),len(B2)
    T=[sum(M[j,i]*B2[j] for j in range(m))for i in range(k)]
    print("The images of basis elements are")
    for i in range(k):
        print(f'T({B1[i]})={T[i]}')
    x=sp.symbols(f'x:{k}')
    c=sp.symbols(f'c:{k}')
    for i in range(k):
        x=x-c[i]*B1[i]
    s=sp.solve(x,c)
    transform=0
    for i,v in enumerate(s):
        transform+=s[v]*T[i]
    print("The required linear transformation is",tuple(transform))

B1=np.array([[1,-1],[1,1]])
B2=np.array([[1,0],[0,1]])
M=sp.Matrix([[2,3],[4,-5]])
lt_of_matrix(B1,B2,M)


Exp 5 : Rank Nullity Theorem
#1
import sympy as sp
def rank_nullity(M,domain_dim):
    rank = M.rank()
    B = M.rref()
    print("The range space is spanned by the nonzero rows of:\n",B[0])
    N = M.transpose().nullspace()
    print("The nullspace is spanned by the colums of \n",N)
    nullity = len(N)
    if domain_dim == rank + nullity:
        print("The rank nullity theorem is verified")
        return True
    else:
        print("Retry again")
        return False
M = sp.Matrix([[1,-1,0],[2,0,1],[1,1,1]])
rank_nullity(M,3)

#2
import sympy as sp
def rank_nullity(M,domain_dim):
    rank = M.rank()
    B = M.rref()
    print("The range space is spanned by the nonzero rows of:\n",B[0])
    N = M.transpose().nullspace()
    print("The nullspace is spanned by the colums of \n",N)
    nullity = len(N)
    if domain_dim == rank + nullity:
        print("The rank nullity theorem is verified")
        return True
    else:
        print("Retry again")
        return False
M = sp.Matrix([[1,1,0],[0,1,1],[1,2,1]])
rank_nullity(M,3)


Exp 6 : Solutions to the problems in different types of PDE

# Type 1

from sympy import *
from sympy.abc import *
f = Function('f')
u = f(x,y)
p = u.diff(x)
q = u.diff(y)
eq = p**2 + q**2 -1
Eq = eq.subs(p,a).subs(q,b)
b_val = solve(Eq,b)
z = a*x + b*y + c
ans = z.subs(b,b_val[0])
print("The requried solution is",ans)




#Type 2 (updated program)

from sympy import *
from sympy.abc import *
z = Function('z')(u)
eq = p*(1-q**2)-q*(1-z) #Change eq function here
eq1 = eq.subs(p,diff(z,u)).subs(q,a*diff(z,u))
sol = dsolve(eq1)
ans = sol[0].subs(u,z+a*y)
pprint(ans)





#Type 3

from sympy import *
from sympy.abc import *
lhs = p*x
rhs = q*y
r1 = Eq(lhs,k)
r2 = Eq(rhs,k)
h1 = solve(r1,p)
print(h1)
h2 = solve(r2,q)
print("p =",h1,"and q =",h2)
z = integrate(h1[0],x) + integrate(h2[0],y)
print("The solution is",z)

#Type 4

from sympy import *
from sympy.abc import *
z = p*x + q*y + (p**2 - q**2)
sol = z.subs([(p,a),(q,b)])
print("General solution is",sol)

Exp 7 : Solutions of the problems of TDE (Only check integrability)
#1
from sympy import *
x,y,z = symbols('x y z')
p = y + z
q = z + x
r = x + y
f = simplify(p*(diff(q,z) - diff(r,y)) + q*(diff(r,x) - diff(p,z)) + r*(diff(p,y) - diff(q,x)))
if f == 0:
    print("Integrable")
else:
    print("Not integrable")

#2
from sympy import *
x,y,z = symbols('x y z')
p = y*z
q = 2*x*z
r = -3*x*y
f = simplify(p*(diff(q,z) - diff(r,y)) + q*(diff(r,x) - diff(p,z)) + r*(diff(p,y) - diff(q,x)))
if f == 0:
    print("Integrable")
else:
    print("Not integrable")

#3
from sympy import *
x,y,z = symbols('x y z')
p = y**2
q = (x+z)**2
r = y**2
f = simplify(p*(diff(q,z) - diff(r,y)) + q*(diff(r,x) - diff(p,z)) + r*(diff(p,y) - diff(q,x)))
if f == 0:
    print("Integrable")
else:
    print("Not integrable")

#4
from sympy import *
x,y,z = symbols('x y z')
p = 3*x**2
q = 3*y**2
r = -(x**3 + y**3 + exp(2*z))
f = simplify(p*(diff(q,z) - diff(r,y)) + q*(diff(r,x) - diff(p,z)) + r*(diff(p,y) - diff(q,x)))
if f == 0:
    print("Integrable")
else:
    print("Not integrable")

#5
from sympy import *
x,y,z = symbols('x y z')
p = 2*x**2 + 2*x*y + 2*x*z**2
q = 1
r = 2*z
f = simplify(p*(diff(q,z) - diff(r,y)) + q*(diff(r,x) - diff(p,z)) + r*(diff(p,y) - diff(q,x)))
if f == 0:
    print("Integrable")
else:
    print("Not integrable")

#6
from sympy import *
x,y,z = symbols('x y z')
p = 2*x + y**2 + 2*x*z
q = 2*x*y
r = x**2
f = simplify(p*(diff(q,z) - diff(r,y)) + q*(diff(r,x) - diff(p,z)) + r*(diff(p,y) - diff(q,x)))
if f == 0:
    print("Integrable")
else:
    print("Not integrable")

#7
from sympy import *
x,y,z = symbols('x y z')
p = y*z
q = -x*z
r = -(2*y + 3*x*z**3)*x
f = simplify(p*(diff(q,z) - diff(r,y)) + q*(diff(r,x) - diff(p,z)) + r*(diff(p,y) - diff(q,x)))
if f == 0:
    print("Integrable")
else:
    print("Not integrable")


Exp 8 : Fourier Series
#1
from sympy import pi,sin,cos,integrate,pprint
from sympy.abc import x,t
def wave(c,l,f,g):
    sol = 0
    for n in range(1,4):
        en = (2/l)*integrate(f*sin(n*pi*x/l),(x,0,l))
        fn = (2/(n*pi*c))*integrate(g*sin(n*pi*x/l),(x,0,l))
        sol = sol + (en*cos(n*pi*c*t/l) + fn*sin(n*pi*c*t/l))*sin(n*pi*x/l)
    print("The solution of the given wave equation is")
    pprint(sol)
wave(1,pi,sin(x),0)

#2
from sympy import pi,sin,cos,integrate,pprint
from sympy.abc import x,t,c
def wave(c,l,f,g):
    sol = 0
    for n in range(1,4):
        en = (2/l)*integrate(f*sin(n*pi*x/l),(x,0,l))
        fn = (2/(n*pi*c))*integrate(g*sin(n*pi*x/l),(x,0,l))
        sol = sol + (en*cos(n*pi*c*t/l) + fn*sin(n*pi*c*t/l))*sin(n*pi*x/l)
    print("The solution of the given wave equation is")
    pprint(sol)
wave(c,2,sin(pi*x/2)**3,0)



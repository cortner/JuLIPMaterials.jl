
module Steve

using JuLIP


# Functions to calculate each of the matrices in the BBS approach.

function CalcQ(c,t,ω)
   c12, c44 = c[1], c[2]

    A1=c[1]-c[2]+2*c[3];
    B1=c[1]+c[2];

    θ=acos(t[3]);
    ϕ=atan2(t[2],t[1]);

    com=cos(ω);
    som=sin(ω);
    cth=cos(θ);
    sth=sin(θ);
    cph=cos(ϕ);
    sph=sin(ϕ);

    n1=-com*cph*cth+som*sph;
    n2=-cph*som-com*cth*sph;
    n3=com*sth;

    nn = [A1*n1^2+c44 B1*n1*n2    B1*n1*n3;
          B1*n1*n2    A1*n2^2+c44 B1*n2*n3;
          B1*n1*n3    B1*n2*n3    A1*n3^2+c44];

    return -inv(nn);   # TODO: switch to pinv
end

function CalcS(c,t,ω)
   c44 = c[2]
   c12 = c[1]

    A1=c[1]-c[2]+2*c[3];
    B1=c[1]+c[2];

    θ=acos(t[3]);
    ϕ=atan2(t[2],t[1]);

    com=cos(ω);
    som=sin(ω);
    cth=cos(θ);
    sth=sin(θ);
    cph=cos(ϕ);
    sph=sin(ϕ);

    n1=-com*cph*cth+som*sph;
    n2=-cph*som-com*cth*sph;
    n3=com*sth;

    m1=cph*cth*som+com*sph;
    m2=-com*cph+cth*som*sph;
    m3=-som*sth;

    nn = [A1*n1^2+c44 B1*n1*n2    B1*n1*n3;
          B1*n1*n2    A1*n2^2+c44 B1*n2*n3;
          B1*n1*n3    B1*n2*n3    A1*n3^2+c44];

    nm = [A1*m1*n1        c12*m2*n1+c44*m1*n2 c12*m3*n1+c44*m1*n3;
          c12*m1*n2+c44*m2*n1 A1*m2*n2            c12*m3*n2+c44*m2*n3;
          c12*m1*n3+c44*m3*n1 c12*m2*n3+c44*m3*n2 A1*m3*n3];

    return -nn\nm;;
end

function CalcB(c,t,ω)
   c12, c44 = c[1], c[2]

A1=c[1]-c[2]+2*c[3];
B1=c[1]+c[2];

θ=acos(t[3]);
ϕ=atan2(t[2],t[1]);

com=cos(ω);
som=sin(ω);
cth=cos(θ);
sth=sin(θ);
cph=cos(ϕ);
sph=sin(ϕ);

n1=-com*cph*cth+som*sph;
n2=-cph*som-com*cth*sph;
n3=com*sth;

m1=cph*cth*som+com*sph;
m2=-com*cph+cth*som*sph;
m3=-som*sth;

# construct (mm) matrix
mm=[A1*m1*m1+c44        c12*m1*m2+c44*m2*m1 c12*m1*m3+c44*m3*m1;
    c12*m1*m2+c44*m2*m1 A1*m2*m2+c44        c12*m2*m3+c44*m3*m2;
    c12*m1*m3+c44*m3*m1 c12*m2*m3+c44*m3*m2 A1*m3*m3+c44];

# construct (nn) matrix
nn = [A1*n1^2+c44 B1*n1*n2    B1*n1*n3;
      B1*n1*n2    A1*n2^2+c44 B1*n2*n3;
      B1*n1*n3    B1*n2*n3    A1*n3^2+c44];

# construct (mn) matrix
mn = [A1*m1*n1            c12*m1*n2+c44*m2*n1 c12*m1*n3+c44*m3*n1;
      c12*m2*n1+c44*m1*n2 A1*m2*n2            c12*m2*n3+c44*m3*n2;
      c12*m3*n1+c44*m1*n3 c12*m3*n2+c44*m2*n3 A1*m3*n3];

nm = mn';

return mm-mn*(nn\nm);
end

"""
`CalcDisp(c,t,b,S::Array{Float64,2},B::Array{Float64,2},xdis,xfld)`

returns 3 component displacement vector u, input arguments are:
c = vector of elastic constants:
t = unit vector of dislocation line direction
b = Burgers' vector
S = matrix S
B = matrix B
xdis = dislocation position (point it passes through, typically (0 0 0))
xfld = field point, in plane perp to dislocation. For t = [111]/sqrt(3),
this will be A(1 1 -2) + B(1,-1,0)
"""
function CalcDisp(c,t,b,S::Array{Float64,2},B::Array{Float64,2}, xdis,
                  xfld::AbstractVector{Float64})

    x = xfld - xdis;

    ω = atan2(x[2], x[1]); # angle from some reference datum
    # NB this is where the non-uniqueness of the displacement field comes in

    Qintegrand = om -> CalcQ(c,t,om)
    Sintegrand = om -> CalcS(c,t,om)
    I1, E1 = quadgk(Qintegrand,0,ω)
    I2, E2 = quadgk(Sintegrand,0,ω)

    # these are the terms in the expression for the displacement in Balluffi about p250 I think!

    dist_2_line = x - dot(x, t) * t
    term1 = -log(norm(dist_2_line))*S;
    term2 = 4π*I1*B;
    term3 = -I2*S;

    return (1/(2π))*(term1+term2+term3)*b;
end


function CalcDisp( Cvoigt, t, b, Xfld::AbstractVector{JVecF};
                   xdis = JVecF(0,0,0) )
   t = t / norm(t)
   c = [Cvoigt[1,2], Cvoigt[4,4], (Cvoigt[1,1] - Cvoigt[1,2]) / 2]
   S = CalcS(c,t,2π) / (2*pi)
   B = CalcB(c,t,2π) / (8*pi*pi)
   return [ JVecF(CalcDisp(c, t, b, S, B, xdis, x)) for x in Xfld ]
end


end

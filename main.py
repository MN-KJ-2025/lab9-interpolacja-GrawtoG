# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from math import cos,pi

def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i (NIE )sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n,int) or n<=0:
        return None
    # xk=[]
    # for k in range(n+1):
    #     xk.append(cos(k*pi/(n-1)))
    xk=np.cos((np.arange(n)*np.pi)/(n-1))
    return xk


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n,int) or n<=0:
        return None
       
    xk=[0.5]
    t=1
    for k in range(n):
        xk.append((t:=-t))

    return xk.append(t*0.5)



def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(xi,np.array) or isinstance(yi,np.array) or isinstance(wi,np.array) or isinstance(x,np.array)):
        return None
    
# n->len(x)
# x->xi
# c->wi
# xx->x
# n = 1000;
# fun = inline(’abs(x)+.5*x-x.ˆ2’);
# x = cos(pi*(0:n)’/n);
# f = fun(x);
# c = [1/2; ones(n-1,1); 1/2].*(-1).ˆ((0:n)’)

#     xx = linspace(-1,1,5000)’;
# numer = zeros(size(xx));
# denom = zeros(size(xx));
# for j = 1:n+1
# xdiff = xx-x(j);
# temp = c(j)./xdiff;
# numer = numer + temp*f(j);
# denom = denom + temp;
# end
# ff = numer./denom;
# plot(x,f,’.’,xx,ff,’-’)
    
    numer = np.zeros(len(x))
    denom = np.zeros(len(x))
    for j in range(1,n+1):
        xdiff = x-xi(j)
        temp = wi(j)/xdiff
        numer = numer + temp*f(j)
        denom = denom + temp
    ff = numer/denom
    
     
def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(xr,int) or isinstance(xr,float) or isinstance(xr,list),isinstance(xr,nd.array) ):
        return None
    if not (isinstance(x,int) or isinstance(x,float) or isinstance(x,list),isinstance(x,nd.array) ):
        return None
    return np.max(np.abs(xr-x))

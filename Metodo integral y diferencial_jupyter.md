# Docencia
# First cell - Import libraries
import numpy as np
from scipy.integrate import ode
import random
import matplotlib.pyplot as plt
%matplotlib inline

# Second cell - Title for Differential Method
print("="*50)
print("MÉTODO DIFERENCIAL")
print("="*50)

# Third cell - Differential Method Functions
def f(t, Ca, n, k):
    """Reaction rate equation: -dCa/dt = k*Ca^n"""
    return -k * (Ca**n)

def generate_and_analyze_data():
    # Parameters
    n_points = 8
    k = random.uniform(0.1, 0.5)  # rate constant
    n = random.randint(1, 2)      # reaction order (1 or 2)
    
    # Initial concentration and time
    Ca0 = 1.0
    t_final = 10.0
    
    # Time increment
    incr_t = t_final / (n_points - 1)
    
    # Initialize arrays
    t = np.zeros(n_points)
    Ca = np.zeros(n_points)
    
    # Set initial conditions
    t[0] = 0
    Ca[0] = Ca0
    
    # Setup ODE solver
    r = ode(f).set_integrator('vode', method='bdf', atol=1.0e-05, rtol=1.0e-05)
    r.set_initial_value(Ca0, 0).set_f_params(n, k)
    
    # Solve ODE
    for i in range(1, n_points):
        r.integrate(r.t + incr_t)
        Ca[i] = r.y[0] + random.uniform(-0.001, 0.001) * r.y[0]  # Add small random noise
        t[i] = r.t
    
    # Print data table with formatted values
    print("\nGenerated Data:")
    print("-------------------------")
    print("Time (min) | Ca (mol/L)")
    print("-------------------------")
    for i in range(len(t)):
        print(f"{t[i]:9.2f} | {Ca[i]:9.2f}")
    print("-------------------------")
    
    return Ca, t, n, k  # Return actual n and k values for comparison

def analyze_differential_method(Ca, t):
    # Calculate rates and logarithms for analysis
    Ca_avg = (Ca[:-1] + Ca[1:]) / 2
    DCa = np.diff(Ca)
    Dt = np.diff(t)
    r = -DCa/Dt
    lnr = np.log(r)
    lnCa_avg = np.log(Ca_avg)
    
    # Perform linear regression
    slope, intercept = np.polyfit(lnCa_avg, lnr, 1)
    
    # Calculate R-squared
    predicted_lnr = slope * lnCa_avg + intercept
    correlation_matrix = np.corrcoef(lnCa_avg, lnr)
    r_squared = correlation_matrix[0,1]**2
    
    # Print intermediate calculations
    print("\nIntermediate Calculations:")
    print("----------------------------------------")
    print("Ca_avg    | Rate (r) | ln(Ca_avg) | ln(r)")
    print("----------------------------------------")
    for i in range(len(Ca_avg)):
        print(f"{Ca_avg[i]:9.2f} | {r[i]:8.2f} | {lnCa_avg[i]:9.2f} | {lnr[i]:6.2f}")
    print("----------------------------------------")
    
    return slope, intercept, r_squared, lnCa_avg, lnr, predicted_lnr

def is_near_integer(slope, tolerance=0.05):
    """Check if slope is within ±tolerance of any integer and is positive"""
    if slope <= 0:  # Check if slope is positive
        return False
    nearest_int = round(slope)
    return abs(slope - nearest_int) <= tolerance
    
# Fourth cell - Generate and Analyze Data (Differential Method)
# Generate data until both conditions are met
max_attempts = 1000
attempt = 0
best_r_squared = 0
best_data = None

while attempt < max_attempts:
    Ca, t, true_n, true_k = generate_and_analyze_data()
    slope, intercept, r_squared, lnCa_avg, lnr, predicted_lnr = analyze_differential_method(Ca, t)
    
    if r_squared >= 0.99 and is_near_integer(slope):
        best_data = (Ca, t, slope, intercept, r_squared, lnCa_avg, lnr, predicted_lnr, true_n, true_k)
        break
    if r_squared > best_r_squared and is_near_integer(slope):
        best_r_squared = r_squared
        best_data = (Ca, t, slope, intercept, r_squared, lnCa_avg, lnr, predicted_lnr, true_n, true_k)
    attempt += 1

if attempt == max_attempts:
    print(f"Could not find data meeting all conditions. Showing best attempt (R² = {best_r_squared:.4f})")
Ca, t, slope, intercept, r_squared, lnCa_avg, lnr, predicted_lnr, true_n, true_k = best_data

# Calculate k from intercept
ln_k = intercept
k = np.exp(ln_k)

print("\nDifferential Method Results:")
print(f"True values: n = {true_n}, k = {true_k:.4f}")
print(f"Calculated values:")
print(f"n (slope) = {slope:.4f}")
print(f"k = {k:.4f}")
print(f"R-squared = {r_squared:.4f}")
print(f"Equation: ln(r) = {slope:.4f}ln(Ca_avg) + {intercept:.4f}")

# Fifth cell - Plot Differential Method Results
plt.figure(figsize=(10, 6))
plt.scatter(lnCa_avg, lnr, color='blue', label='Data points')
plt.plot(lnCa_avg, predicted_lnr, color='red', label='Regression line')
plt.xlabel('ln(Ca_avg)')
plt.ylabel('ln(r)')
plt.title('Differential Method: ln(r) vs ln(Ca_avg)')

# Add equation, R², and k value to the plot
equation = f'ln(r) = {slope:.4f}ln(Ca_avg) + {intercept:.4f}'
r_squared_text = f'R² = {r_squared:.4f}'
k_text = f'k = {k:.4f}'
plt.text(0.05, 0.95, equation + '\n' + r_squared_text + '\n' + k_text, 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

plt.legend()
plt.grid(True)
plt.show()

# Sixth cell - Title for Integral Method
print("="*50)
print("MÉTODO INTEGRAL")
print("="*50)

# Seventh cell - Integral Method Functions
def analyze_integral_method(Ca, t, n_guess):
    """
    Analyze data using integral method for nth order reaction
    Returns: k, R², y, y_pred
    """
    Ca0 = Ca[0]
    
    if n_guess == 1:
        # First order: ln(Ca/Ca0) = -kt
        y = np.log(Ca/Ca0)
        slope, intercept = np.polyfit(t, y, 1)
        k = -slope
        y_pred = slope * t + intercept
        
    elif n_guess == 2:  # Fixed syntax error here
        # Second order: 1/Ca - 1/Ca0 = kt
        y = 1/Ca - 1/Ca0
        slope, intercept = np.polyfit(t, y, 1)
        k = slope
        y_pred = slope * t + intercept
        
    else:  # n_guess == 3
        # Third order: 1/(2*Ca^2) - 1/(2*Ca0^2) = kt
        y = 1/(2*Ca**2) - 1/(2*Ca0**2)
        slope, intercept = np.polyfit(t, y, 1)
        k = slope
        y_pred = slope * t + intercept
    
    # Calculate R-squared
    correlation_matrix = np.corrcoef(t, y)
    r_squared = correlation_matrix[0,1]**2
    
    return k, r_squared, y, y_pred

# Test first, second and third order
k1, r_squared1, y1, y_pred1 = analyze_integral_method(Ca, t, 1)
k2, r_squared2, y2, y_pred2 = analyze_integral_method(Ca, t, 2)
k3, r_squared3, y3, y_pred3 = analyze_integral_method(Ca, t, 3)  # Fixed variable name here

print("\nIntegral Method Results:")
print("\nFirst Order Test:")
print(f"k = {k1:.4f}")
print(f"R² = {r_squared1:.4f}")

print("\nSecond Order Test:")
print(f"k = {k2:.4f}")
print(f"R² = {r_squared2:.4f}")

print("\nThird Order Test:")  # Fixed typo here
print(f"k = {k3:.4f}")
print(f"R² = {r_squared3:.4f}")  # Fixed variable name here

# Determine best fit among all three orders
r_squared_values = [r_squared1, r_squared2, r_squared3]
k_values = [k1, k2, k3]
y_values = [y1, y2, y3]
y_pred_values = [y_pred1, y_pred2, y_pred3]
ylabels = ['ln(Ca/Ca0)', '1/Ca - 1/Ca0', '1/(2Ca²) - 1/(2Ca0²)']

best_index = np.argmax(r_squared_values)
best_order = best_index + 1
best_k = k_values[best_index]
best_r_squared = r_squared_values[best_index]
y = y_values[best_index]
y_pred = y_pred_values[best_index]
ylabel = ylabels[best_index]

print(f"\nBest fit: {best_order}{'st' if best_order == 1 else 'nd' if best_order == 2 else 'rd'} Order")
print(f"k = {best_k:.4f}")
print(f"R² = {best_r_squared:.4f}")

# Print data table
print("\nIntegral Method Data:")
print("----------------------------------------")
print("Time (min) | Ca (mol/L) | y")
print("----------------------------------------")
for i in range(len(t)):
    print(f"{t[i]:9.2f} | {Ca[i]:9.2f} | {y[i]:9.2f}")
print("----------------------------------------")


# Eighth cell - Plot Integral Method Results
plt.figure(figsize=(10, 6))
plt.scatter(t, y, color='blue', label='Data points')
plt.plot(t, y_pred, color='red', label='Regression line')
plt.xlabel('Time')
plt.ylabel(ylabel)
plt.title(f'Integral Method: {best_order}st Order Reaction')

# Add equation and R² to the plot
equation = f'k = {best_k:.4f}'
r_squared_text = f'R² = {best_r_squared:.4f}'
plt.text(0.05, 0.95, equation + '\n' + r_squared_text, 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

plt.legend()
plt.grid(True)
plt.show()

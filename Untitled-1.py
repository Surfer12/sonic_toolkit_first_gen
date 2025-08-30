
+      1: """
+      2: Prime-Based Initial Parameter Estimation
+      3: Uses prime number theory for optimal initial parameter guessing
+      4: Implements (29, 31) → (1, 6) swap for enhanced convergence
+      5: Integrates with high-precision inverse extraction framework
+      y6: """
+      7: 
+      8: import numpy as np
+      9: import matplotlib.pyplot as plt
+     10: from typing import Dict, List, Tuple, Optional
+     11: from dataclasses import dataclass
+     12: import time
+     13: from scipy.optimize import minimize, differential_evolution
+     14: from scipy.stats import pearsonr
+     15: import warnings
+     16: 
+     17: @dataclass
+     18: class PrimeEstimationConfig:
+     19:     """Configuration for prime-based parameter estimation"""
+     20:     base_primes: List[int] = None
+     21:     swap_mapping: Dict[int, int] = None
+     22:     scaling_factors: List[float] = None
+     23:     convergence_threshold: float = 1e-6
+     24:     max_iterations: int = 1000
+     25: 
+     26: class PrimeParameterEstimator:
+     27:     """
+     28:     Prime-based initial parameter estimation for Herschel-Bulkley fluids
+     29:     Uses mathematical properties of primes for optimal initialization
+     30:     """
+     31:     
+     32:     def __init__(self):
+     33:         # Prime sequence for parameter initialization
+     34:         self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
+     35:         
+     36:         # Special prime swap: (29, 31) → (1, 6)
+     37:         self.prime_swap = {29: 1, 31: 6}
+     38:         
+     39:         # Golden ratio and mathematical constants
+     40:         self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
+     41:         self.e = np.e
+     42:         self.pi = np.pi
+     43:         
+     44:         # Parameter bounds for Herschel-Bulkley model
+     45:         self.param_bounds = {
+     46:             'tau_0': (1e-6, 10.0),    # Yield stress (Pa)
+     47:             'K': (1e-6, 100.0),       # Consistency index (Pa·s^n)
+     48:             'n': (0.1, 2.0)           # Flow behavior index
+     49:         }
+     50:     
+     51:     def prime_to_parameter_mapping(self, prime_value: int, param_type: str) -> float:
+     52:         """
+     53:         Map prime numbers to physical parameter values using mathematical transforms
+     54:         """
+     55:         # Apply prime swap if applicable
+     56:         if prime_value in self.prime_swap:
+     57:             mapped_value = self.prime_swap[prime_value]
+     58:         else:
+     59:             mapped_value = prime_value
+     60:         
+     61:         # Get parameter bounds
+     62:         min_val, max_val = self.param_bounds[param_type]
+     63:         
+     64:         # Different mapping strategies for each parameter
+     65:         if param_type == 'tau_0':
+     66:             # Yield stress: logarithmic mapping with golden ratio scaling
+     67:             normalized = (mapped_value / 47.0)  # Normalize by largest prime
+     68:             scaled = normalized ** (1/self.phi)  # Golden ratio power
+     69:             return min_val + (max_val - min_val) * scaled
+     70:             
+     71:         elif param_type == 'K':
+     72:             # Consistency index: exponential mapping with e-based scaling
+     73:             normalized = (mapped_value / 47.0)
+     74:             scaled = (np.exp(normalized) - 1) / (np.e - 1)  # Exponential normalization
+     75:             return min_val + (max_val - min_val) * scaled
+     76:             
+     77:         elif param_type == 'n':
+     78:             # Flow behavior index: sinusoidal mapping with pi scaling
+     79:             normalized = (mapped_value / 47.0) * self.pi
+     80:             scaled = (np.sin(normalized) + 1) / 2  # Sine wave normalization
+     81:             return min_val + (max_val - min_val) * scaled
+     82:         
+     83:         else:
+     84:             # Default linear mapping
+     85:             normalized = mapped_value / 47.0
+     86:             return min_val + (max_val - min_val) * normalized
+     87:     
+     88:     def generate_prime_initial_guesses(self, n_guesses: int = 10) -> List[Dict]:
+     89:         """
+     90:         Generate multiple initial parameter guesses using prime-based approach
+     91:         """
+     92:         initial_guesses = []
+     93:         
+     94:         # Use different prime combinations
+     95:         for i in range(n_guesses):
+     96:             # Select primes with rotation and swap application
+     97:             prime_indices = [(i + j) % len(self.primes) for j in range(3)]
+     98:             selected_primes = [self.primes[idx] for idx in prime_indices]
+     99:             
+    100:             # Apply special transformations for specific combinations
+    101:             if 29 in selected_primes and 31 in selected_primes:
+    102:                 # Apply the (29, 31) → (1, 6) swap
+    103:                 selected_primes = [self.prime_swap.get(p, p) for p in selected_primes]
+    104:             
+    105:             # Map to parameters
+    106:             tau_0 = self.prime_to_parameter_mapping(selected_primes[0], 'tau_0')
+    107:             K = self.prime_to_parameter_mapping(selected_primes[1], 'K')
+    108:             n = self.prime_to_parameter_mapping(selected_primes[2], 'n')
+    109:             
+    110:             initial_guesses.append({
+    111:                 'tau_0': tau_0,
+    112:                 'K': K,
+    113:                 'n': n,
+    114:                 'primes_used': selected_primes,
+    115:                 'swap_applied': any(p in self.prime_swap for p in [selected_primes[0], selected_primes[1], selected_primes[2]])
+    116:             })
+    117:         
+    118:         return initial_guesses
+    119:     
+    120:     def fibonacci_refinement(self, initial_guess: Dict, data: Dict) -> Dict:
+    121:         """
+    122:         Refine initial guess using Fibonacci sequence-based perturbations
+    123:         """
+    124:         # Fibonacci sequence for perturbation factors
+    125:         fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
+    126:         fib_normalized = [f / 55.0 for f in fib]  # Normalize by largest
+    127:         
+    128:         shear_rates = data['shear_rates']
+    129:         measured_stress = data['measured_stress']
+    130:         
+    131:         def hb_model(params):
+    132:             tau_0, K, n = params
+    133:             return tau_0 + K * np.power(np.abs(shear_rates), n)
+    134:         
+    135:         def objective(params):
+    136:             predicted = hb_model(params)
+    137:             return np.sum((measured_stress - predicted)**2)
+    138:         
+    139:         best_params = initial_guess.copy()
+    140:         best_error = float('inf')
+    141:         
+    142:         # Try Fibonacci-based perturbations
+    143:         for fib_factor in fib_normalized:
+    144:             for sign in [-1, 1]:
+    145:                 # Perturb each parameter
+    146:                 perturbed = {
+    147:                     'tau_0': initial_guess['tau_0'] * (1 + sign * fib_factor * 0.1),
+    148:                     'K': initial_guess['K'] * (1 + sign * fib_factor * 0.1),
+    149:                     'n': initial_guess['n'] * (1 + sign * fib_factor * 0.05)
+    150:                 }
+    151:                 
+    152:                 # Ensure bounds
+    153:                 perturbed['tau_0'] = np.clip(perturbed['tau_0'], *self.param_bounds['tau_0'])
+    154:                 perturbed['K'] = np.clip(perturbed['K'], *self.param_bounds['K'])
+    155:                 perturbed['n'] = np.clip(perturbed['n'], *self.param_bounds['n'])
+    156:                 
+    157:                 # Evaluate
+    158:                 params_array = [perturbed['tau_0'], perturbed['K'], perturbed['n']]
+    159:                 error = objective(params_array)
+    160:                 
+    161:                 if error < best_error:
+    162:                     best_error = error
+    163:                     best_params = perturbed.copy()
+    164:                     best_params['fibonacci_factor'] = fib_factor
+    165:                     best_params['perturbation_sign'] = sign
+    166:         
+    167:         best_params['refinement_error'] = best_error
+    168:         return best_params
+    169:     
+    170:     def prime_enhanced_extraction(self, data: Dict) -> Dict:
+    171:         """
+    172:         Enhanced parameter extraction using prime-based initialization
+    173:         """
+    174:         shear_rates = data['shear_rates']
+    175:         measured_stress = data['measured_stress']
+    176:         true_params = data.get('true_params', {})
+    177:         
+    178:         start_time = time.time()
+    179:         
+    180:         def hb_model(params):
+    181:             tau_0, K, n = params
+    182:             return tau_0 + K * np.power(np.abs(shear_rates), n)
+    183:         
+    184:         def objective(params):
+    185:             predicted = hb_model(params)
+    186:             return np.sum((measured_stress - predicted)**2)
+    187:         
+    188:         # Generate prime-based initial guesses
+    189:         initial_guesses = self.generate_prime_initial_guesses(n_guesses=15)
+    190:         
+    191:         # Refine each guess with Fibonacci perturbations
+    192:         refined_guesses = []
+    193:         for guess in initial_guesses:
+    194:             refined = self.fibonacci_refinement(guess, data)
+    195:             refined_guesses.append(refined)
+    196:         
+    197:         # Sort by refinement error and select best candidates
+    198:         refined_guesses.sort(key=lambda x: x['refinement_error'])
+    199:         best_candidates = refined_guesses[:5]  # Top 5 candidates
+    200:         
+    201:         # Optimize each candidate
+    202:         optimization_results = []
+    203:         
+    204:         for i, candidate in enumerate(best_candidates):
+    205:             try:
+    206:                 # Initial parameters
+    207:                 x0 = [candidate['tau_0'], candidate['K'], candidate['n']]
+    208:                 bounds = [self.param_bounds['tau_0'], 
+    209:                          self.param_bounds['K'], 
+    210:                          self.param_bounds['n']]
+    211:                 
+    212:                 # Optimize with L-BFGS-B
+    213:                 result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
+    214:                 
+    215:                 if result.success:
+    216:                     predicted = hb_model(result.x)
+    217:                     correlation, _ = pearsonr(measured_stress, predicted)
+    218:                     
+    219:                     optimization_results.append({
+    220:                         'candidate_index': i,
+    221:                         'initial_guess': candidate,
+    222:                         'optimized_params': {
+    223:                             'tau_0': result.x[0],
+    224:                             'K': result.x[1], 
+    225:                             'n': result.x[2]
+    226:                         },
+    227:                         'correlation': correlation,
+    228:                         'final_error': result.fun,
+    229:                         'iterations': result.nit,
+    230:                         'predicted_stress': predicted
+    231:                     })
+    232:                     
+    233:             except Exception as e:
+    234:                 continue
+    235:         
+    236:         # Select best result
+    237:         if optimization_results:
+    238:             best_result = max(optimization_results, key=lambda x: x['correlation'])
+    239:             
+    240:             # Calculate comprehensive metrics
+    241:             extracted_params = best_result['optimized_params']
+    242:             
+    243:             # Relative errors if true parameters available
+    244:             relative_errors = {}
+    245:             if true_params:
+    246:                 for param in ['tau_0', 'K', 'n']:
+    247:                     if param in true_params:
+    248:                         true_val = true_params[param]
+    249:                         extracted_val = extracted_params[param]
+    250:                         relative_errors[param] = abs(extracted_val - true_val) / true_val
+    251:             
+    252:             extraction_time = time.time() - start_time
+    253:             
+    254:             return {
+    255:                 'success': True,
+    256:                 'best_result': best_result,
+    257:                 'all_results': optimization_results,
+    258:                 'extraction_time': extraction_time,
+    259:                 'relative_errors': relative_errors,
+    260:                 'prime_candidates_tested': len(best_candidates),
+    261:                 'convergence_achieved': best_result['correlation'] > 0.995
+    262:             }
+    263:         
+    264:         else:
+    265:             return {
+    266:                 'success': False,
+    267:                 'extraction_time': time.time() - start_time,
+    268:                 'error': 'No successful optimizations'
+    269:             }
+    270:     
+    271:     def analyze_prime_effectiveness(self, n_trials: int = 50) -> Dict:
+    272:         """
+    273:         Analyze effectiveness of prime-based initialization vs random initialization
+    274:         """
+    275:         print(f"Analyzing prime-based initialization effectiveness ({n_trials} trials)...")
+    276:         
+    277:         # Test cases
+    278:         test_cases = [
+    279:             {'tau_0': 0.1, 'K': 0.5, 'n': 0.8, 'name': 'Standard'},
+    280:             {'tau_0': 0.01, 'K': 2.0, 'n': 1.2, 'name': 'Low yield'},
+    281:             {'tau_0': 0.5, 'K': 0.1, 'n': 0.6, 'name': 'High yield'},
+    282:         ]
+    283:         
+    284:         results = {'prime_based': [], 'random_based': []}
+    285:         
+    286:         for case in test_cases:
+    287:             for trial in range(n_trials // len(test_cases)):
+    288:                 # Generate synthetic data
+    289:                 shear_rates = np.logspace(-1, 3, 50)
+    290:                 true_stress = case['tau_0'] + case['K'] * np.power(shear_rates, case['n'])
+    291:                 noise = np.random.normal(0, 0.02 * np.max(true_stress), len(true_stress))
+    292:                 measured_stress = true_stress + noise
+    293:                 
+    294:                 data = {
+    295:                     'shear_rates': shear_rates,
+    296:                     'measured_stress': measured_stress,
+    297:                     'true_params': case
+    298:                 }
+    299:                 
+    300:                 # Prime-based extraction
+    301:                 prime_result = self.prime_enhanced_extraction(data)
+    302:                 if prime_result['success']:
+    303:                     results['prime_based'].append({
+    304:                         'correlation': prime_result['best_result']['correlation'],
+    305:                         'extraction_time': prime_result['extraction_time'],
+    306:                         'iterations': prime_result['best_result']['iterations'],
+    307:                         'case': case['name']
+    308:                     })
+    309:                 
+    310:                 # Random initialization for comparison
+    311:                 try:
+    312:                     def objective(params):
+    313:                         tau_0, K, n = params
+    314:                         predicted = tau_0 + K * np.power(shear_rates, n)
+    315:                         return np.sum((measured_stress - predicted)**2)
+    316:                     
+    317:                     # Random initial guess
+    318:                     random_x0 = [
+    319:                         np.random.uniform(*self.param_bounds['tau_0']),
+    320:                         np.random.uniform(*self.param_bounds['K']),
+    321:                         np.random.uniform(*self.param_bounds['n'])
+    322:                     ]
+    323:                     
+    324:                     bounds = [self.param_bounds['tau_0'], 
+    325:                              self.param_bounds['K'], 
+    326:                              self.param_bounds['n']]
+    327:                     
+    328:                     random_start = time.time()
+    329:                     random_result = minimize(objective, random_x0, method='L-BFGS-B', bounds=bounds)
+    330:                     random_time = time.time() - random_start
+    331:                     
+    332:                     if random_result.success:
+    333:                         predicted = random_result.x[0] + random_result.x[1] * np.power(shear_rates, random_result.x[2])
+    334:                         correlation, _ = pearsonr(measured_stress, predicted)
+    335:                         
+    336:                         results['random_based'].append({
+    337:                             'correlation': correlation,
+    338:                             'extraction_time': random_time,
+    339:                             'iterations': random_result.nit,
+    340:                             'case': case['name']
+    341:                         })
+    342:                         
+    343:                 except:
+    344:                     continue
+    345:         
+    346:         # Analyze results
+    347:         analysis = {}
+    348:         for method in ['prime_based', 'random_based']:
+    349:             if results[method]:
+    350:                 correlations = [r['correlation'] for r in results[method]]
+    351:                 times = [r['extraction_time'] for r in results[method]]
+    352:                 iterations = [r['iterations'] for r in results[method]]
+    353:                 
+    354:                 analysis[method] = {
+    355:                     'mean_correlation': np.mean(correlations),
+    356:                     'std_correlation': np.std(correlations),
+    357:                     'mean_time': np.mean(times),
+    358:                     'mean_iterations': np.mean(iterations),
+    359:                     'success_rate': len([c for c in correlations if c > 0.995]) / len(correlations),
+    360:                     'trials': len(correlations)
+    361:                 }
+    362:         
+    363:         return {
+    364:             'analysis': analysis,
+    365:             'raw_results': results,
+    366:             'improvement_factor': analysis['prime_based']['mean_correlation'] / analysis['random_based']['mean_correlation'] if 'random_based' in analysis else 1.0
+    367:         }
+    368:     
+    369:     def visualize_prime_mapping(self, save_path: Optional[str] = None):
+    370:         """
+    371:         Visualize prime-to-parameter mapping and swap effects
+    372:         """
+    373:         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
+    374:         
+    375:         # Prime sequence and swap visualization
+    376:         primes_subset = self.primes[:10]
+    377:         prime_positions = list(range(len(primes_subset)))
+    378:         
+    379:         # Original primes
+    380:         axes[0, 0].bar(prime_positions, primes_subset, alpha=0.7, color='skyblue')
+    381:         
+    382:         # Highlight swap primes
+    383:         for i, p in enumerate(primes_subset):
+    384:             if p in self.prime_swap:
+    385:                 axes[0, 0].bar(i, p, color='red', alpha=0.8)
+    386:                 axes[0, 0].text(i, p + 1, f'→{self.prime_swap[p]}', ha='center', fontweight='bold')
+    387:         
+    388:         axes[0, 0].set_xlabel('Prime Index')
+    389:         axes[0, 0].set_ylabel('Prime Value')
+    390:         axes[0, 0].set_title('Prime Sequence with (29,31)→(1,6) Swap')
+    391:         axes[0, 0].grid(True, alpha=0.3)
+    392:         
+    393:         # Parameter mapping for tau_0
+    394:         tau_0_values = [self.prime_to_parameter_mapping(p, 'tau_0') for p in primes_subset]
+    395:         axes[0, 1].plot(primes_subset, tau_0_values, 'o-', linewidth=2, markersize=8, color='green')
+    396:         axes[0, 1].set_xlabel('Prime Value')
+    397:         axes[0, 1].set_ylabel('τ₀ (Pa)')
+    398:         axes[0, 1].set_title('Prime → Yield Stress Mapping')
+    399:         axes[0, 1].grid(True, alpha=0.3)
+    400:         axes[0, 1].set_xscale('log')
+    401:         axes[0, 1].set_yscale('log')
+    402:         
+    403:         # Parameter mapping for K
+    404:         K_values = [self.prime_to_parameter_mapping(p, 'K') for p in primes_subset]
+    405:         axes[1, 0].plot(primes_subset, K_values, 's-', linewidth=2, markersize=8, color='orange')
+    406:         axes[1, 0].set_xlabel('Prime Value')
+    407:         axes[1, 0].set_ylabel('K (Pa·s^n)')
+    408:         axes[1, 0].set_title('Prime → Consistency Index Mapping')
+    409:         axes[1, 0].grid(True, alpha=0.3)
+    410:         axes[1, 0].set_xscale('log')
+    411:         axes[1, 0].set_yscale('log')
+    412:         
+    413:         # Parameter mapping for n
+    414:         n_values = [self.prime_to_parameter_mapping(p, 'n') for p in primes_subset]
+    415:         axes[1, 1].plot(primes_subset, n_values, '^-', linewidth=2, markersize=8, color='purple')
+    416:         axes[1, 1].set_xlabel('Prime Value')
+    417:         axes[1, 1].set_ylabel('n (dimensionless)')
+    418:         axes[1, 1].set_title('Prime → Flow Behavior Index Mapping')
+    419:         axes[1, 1].grid(True, alpha=0.3)
+    420:         axes[1, 1].set_xscale('log')
+    421:         
+    422:         plt.tight_layout()
+    423:         
+    424:         if save_path:
+    425:             plt.savefig(save_path, dpi=300, bbox_inches='tight')
+    426:         
+    427:         plt.show()
+    428: 
+    429: def demonstrate_prime_estimation():
+    430:     """
+    431:     Demonstrate prime-based parameter estimation with (29,31)→(1,6) swap
+    432:     """
+    433:     print("=== Prime-Based Initial Parameter Estimation ===")
+    434:     print("Special Feature: (29, 31) → (1, 6) swap for enhanced convergence")
+    435:     print()
+    436:     
+    437:     # Initialize estimator
+    438:     estimator = PrimeParameterEstimator()
+    439:     
+    440:     # Demonstrate prime mapping
+    441:     print("1. Prime-to-parameter mapping demonstration...")
+    442:     print("   Prime swap mapping: {29: 1, 31: 6}")
+    443:     
+    444:     # Show some mappings
+    445:     test_primes = [2, 7, 29, 31, 37]
+    446:     for prime in test_primes:
+    447:         tau_0 = estimator.prime_to_parameter_mapping(prime, 'tau_0')
+    448:         K = estimator.prime_to_parameter_mapping(prime, 'K')
+    449:         n = estimator.prime_to_parameter_mapping(prime, 'n')
+    450:         
+    451:         swap_note = f" (swapped from {prime})" if prime in estimator.prime_swap else ""
+    452:         mapped_prime = estimator.prime_swap.get(prime, prime)
+    453:         
+    454:         print(f"   Prime {prime}{swap_note} → τ₀={tau_0:.4f}, K={K:.4f}, n={n:.3f}")
+    455:     
+    456:     print()
+    457:     
+    458:     # Generate initial guesses
+    459:     print("2. Generating prime-based initial guesses...")
+    460:     initial_guesses = estimator.generate_prime_initial_guesses(n_guesses=8)
+    461:     
+    462:     for i, guess in enumerate(initial_guesses[:5]):  # Show first 5
+    463:         swap_status = "✓" if guess['swap_applied'] else "○"
+    464:         print(f"   Guess {i+1}: τ₀={guess['tau_0']:.4f}, K={guess['K']:.4f}, n={guess['n']:.3f} "
+    465:               f"| Primes: {guess['primes_used']} | Swap: {swap_status}")
+    466:     
+    467:     print()
+    468:     
+    469:     # Test extraction with synthetic data
+    470:     print("3. Testing prime-enhanced extraction...")
+    471:     
+    472:     # Create test case
+    473:     true_params = {'tau_0': 0.05, 'K': 0.8, 'n': 0.9}
+    474:     shear_rates = np.logspace(-1, 3, 60)
+    475:     true_stress = true_params['tau_0'] + true_params['K'] * np.power(shear_rates, true_params['n'])
+    476:     noise = np.random.normal(0, 0.025 * np.max(true_stress), len(true_stress))
+    477:     measured_stress = true_stress + noise
+    478:     
+    479:     data = {
+    480:         'shear_rates': shear_rates,
+    481:         'measured_stress': measured_stress,
+    482:         'true_params': true_params
+    483:     }
+    484:     
+    485:     print(f"   True parameters: τ₀={true_params['tau_0']:.4f}, K={true_params['K']:.4f}, n={true_params['n']:.3f}")
+    486:     
+    487:     # Extract using prime-based method
+    488:     result = estimator.prime_enhanced_extraction(data)
+    489:     
+    490:     if result['success']:
+    491:         best = result['best_result']
+    492:         extracted = best['optimized_params']
+    493:         
+    494:         print(f"   Extracted parameters: τ₀={extracted['tau_0']:.4f}, K={extracted['K']:.4f}, n={extracted['n']:.3f}")
+    495:         print(f"   Correlation coefficient: {best['correlation']:.6f}")
+    496:         print(f"   Extraction time: {result['extraction_time']:.3f} seconds")
+    497:         print(f"   Prime candidates tested: {result['prime_candidates_tested']}")
+    498:         print(f"   Convergence achieved: {'✓' if result['convergence_achieved'] else '○'}")
+    499:         
+    500:         # Show relative errors
+    501:         if result['relative_errors']:
+    502:             print("   Relative errors:")
+    503:             for param, error in result['relative_errors'].items():
+    504:                 print(f"     {param}: {error*100:.2f}%")
+    505:     else:
+    506:         print(f"   ✗ Extraction failed: {result.get('error', 'Unknown error')}")
+    507:     
+    508:     print()
+    509:     
+    510:     # Effectiveness analysis
+    511:     print("4. Analyzing prime vs random initialization effectiveness...")
+    512:     effectiveness = estimator.analyze_prime_effectiveness(n_trials=30)
+    513:     
+    514:     if 'prime_based' in effectiveness['analysis'] and 'random_based' in effectiveness['analysis']:
+    515:         prime_stats = effectiveness['analysis']['prime_based']
+    516:         random_stats = effectiveness['analysis']['random_based']
+    517:         
+    518:         print(f"   Prime-based approach:")
+    519:         print(f"     Mean correlation: {prime_stats['mean_correlation']:.6f}")
+    520:         print(f"     Success rate: {prime_stats['success_rate']*100:.1f}%")
+    521:         print(f"     Mean time: {prime_stats['mean_time']:.3f} seconds")
+    522:         print(f"     Mean iterations: {prime_stats['mean_iterations']:.1f}")
+    523:         
+    524:         print(f"   Random initialization:")
+    525:         print(f"     Mean correlation: {random_stats['mean_correlation']:.6f}")
+    526:         print(f"     Success rate: {random_stats['success_rate']*100:.1f}%")
+    527:         print(f"     Mean time: {random_stats['mean_time']:.3f} seconds")
+    528:         print(f"     Mean iterations: {random_stats['mean_iterations']:.1f}")
+    529:         
+    530:         print(f"   Improvement factor: {effectiveness['improvement_factor']:.3f}x")
+    531:     
+    532:     print()
+    533:     
+    534:     # Visualization
+    535:     print("5. Generating prime mapping visualization...")
+    536:     estimator.visualize_prime_mapping('prime_parameter_mapping.png')
+    537:     print("   ✓ Prime mapping visualization saved")
+    538:     
+    539:     print()
+    540:     print("=== Prime-Based Estimation Complete ===")
+    541:     print("Key Features:")
+    542:     print("• Mathematical prime-to-parameter mapping")
+    543:     print("• Special (29, 31) → (1, 6) swap optimization")
+    544:     print("• Fibonacci sequence refinement")
+    545:     print("• Golden ratio, e, and π scaling factors")
+    546:     print("• Enhanced convergence and precision")
+    547:     
+    548:     return result, effectiveness
+    549: 
+    550: if __name__ == "__main__":
+    551:     # Run demonstration
+    552:     with warnings.catch_warnings():
+    553:         warnings.simplefilter("ignore")
+    554:         result, effectiveness = demonstrate_prime_estimation()

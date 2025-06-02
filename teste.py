import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

class Solution:
    def __init__(self, solution):
        self.solution = solution
        self.fitness = None
        self.f1 = None
        self.f2 = None

def load_data():
    equipamentos = pd.read_csv('Dados/EquipDB.csv', header=None, 
                             names=["ID", "t0", "cluster", "custo_falha"])
    planos = pd.read_csv('Dados/MPDB.csv', header=None, 
                        names=["ID", "k", "custo"]).set_index("ID")
    modelo_falha = pd.read_csv('Dados/ClusterDB.csv', header=None, 
                             names=["ID", "eta", "beta"]).set_index("ID")
    
    dados = {
        'equipamentos': equipamentos,
        'planos': planos,
        'modelo_falha': modelo_falha,
        'n': len(equipamentos),
        'custo_por_plano': planos["custo"].to_dict()
    }
    return dados

def initial_solution(dados, delta_t=5):
    n = dados['n']
    t0 = dados['equipamentos']['t0'].values
    clusters = dados['equipamentos']['cluster'].values
    custo_falha = dados['equipamentos']['custo_falha'].values
    eta = dados['modelo_falha'].loc[clusters, 'eta'].values
    beta = dados['modelo_falha'].loc[clusters, 'beta'].values

    def weibull_cdf(t, eta, beta):
        return 1 - np.exp(-(t / eta)**beta)
    
    Fi_t0 = weibull_cdf(t0, eta, beta)
    Fi_t0_dt = weibull_cdf(t0 + delta_t, eta, beta)
    pi = (Fi_t0_dt - Fi_t0) / (1 - Fi_t0 + 1e-10)
    
    custo_beneficio = pi * custo_falha
    indices_ordenados = np.argsort(custo_beneficio)
    
    sol = np.ones(n, dtype=int)
    terc = n // 3
    sol[indices_ordenados[:terc]] = 1
    sol[indices_ordenados[terc:2*terc]] = 2
    sol[indices_ordenados[2*terc:]] = 3
    
    return sol

def calculate_f1(solution, dados):
    return sum(dados['custo_por_plano'].get(p, 0) for p in solution.solution)

def calculate_f2(solution, dados, delta_t=5):
    t0 = dados['equipamentos']['t0'].values
    clusters = dados['equipamentos']['cluster'].values
    custo_falha = dados['equipamentos']['custo_falha'].values
    planos_k = np.array([dados['planos'].loc[p, 'k'] for p in solution.solution])
    
    eta = dados['modelo_falha'].loc[clusters, 'eta'].values
    beta = dados['modelo_falha'].loc[clusters, 'beta'].values
    
    def weibull_cdf(t, eta, beta):
        return 1 - np.exp(-(t / eta)**beta)
    
    Fi_t0 = weibull_cdf(t0, eta, beta)
    Fi_t0_kdt = weibull_cdf(t0 + planos_k * delta_t, eta, beta)
    
    pi = (Fi_t0_kdt - Fi_t0) / (1 - Fi_t0 + 1e-10)
    return np.sum(pi * custo_falha)

def shake_swap(solution, r=1, planos=3):
    new_solution = copy.deepcopy(solution)
    indices = np.random.choice(len(new_solution.solution), r, replace=False)
    for idx in indices:
        options = [p for p in range(1, planos+1) if p != new_solution.solution[idx]]
        new_solution.solution[idx] = np.random.choice(options)
    return new_solution

def shake_increase(solution, r=1, planos=3):
    new_solution = copy.deepcopy(solution)
    indices = np.random.choice(len(new_solution.solution), r, replace=False)
    for idx in indices:
        if new_solution.solution[idx] < planos:
            new_solution.solution[idx] += 1
    return new_solution

def shake_decrease(solution, r=1):
    new_solution = copy.deepcopy(solution)
    indices = np.random.choice(len(new_solution.solution), r, replace=False)
    for idx in indices:
        if new_solution.solution[idx] > 1:
            new_solution.solution[idx] -= 1
    return new_solution

def local_search(solution, neighborhoods, fobj, dados, max_evaluations):
    current = copy.deepcopy(solution)
    current.fitness = fobj(current, dados)
    evaluations = 1
    
    improved = True
    while improved and evaluations < max_evaluations:
        improved = False
        for neighborhood in neighborhoods:
            neighbor = neighborhood(current)
            neighbor.fitness = fobj(neighbor, dados)
            evaluations += 1
            
            if neighbor.fitness < current.fitness:
                current = neighbor
                improved = True
                break
                
    return current, evaluations

def gvns(initial_solution_func, fobj, max_iter, k_max, neighborhoods, dados, r=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    current = Solution(initial_solution_func())
    current.fitness = fobj(current, dados)
    best = copy.deepcopy(current)
    
    evaluations = 1
    while evaluations < max_iter:
        k = 0
        while k < k_max and evaluations < max_iter:
            shaken = neighborhoods[k](current, r)
            shaken.fitness = fobj(shaken, dados)
            evaluations += 1
            
            local_opt, evals = local_search(shaken, neighborhoods, fobj, dados, max_iter - evaluations)
            evaluations += evals
            
            if local_opt.fitness < best.fitness:
                best = copy.deepcopy(local_opt)
                current = copy.deepcopy(local_opt)
                k = 0
            else:
                k += 1
                
    return best, evaluations

def find_extremes(dados, max_iter=1000):
    neighborhoods = [shake_swap, shake_increase, shake_decrease]
    
    def f1_obj(solution, dados):
        solution.f1 = calculate_f1(solution, dados)
        solution.fitness = solution.f1
        return solution.fitness
    
    best_f1, _ = gvns(lambda: initial_solution(dados), f1_obj, max_iter, len(neighborhoods), neighborhoods, dados)
    
    def f2_obj(solution, dados):
        solution.f2 = calculate_f2(solution, dados)
        solution.fitness = solution.f2
        return solution.fitness
    
    best_f2, _ = gvns(lambda: initial_solution(dados), f2_obj, max_iter, len(neighborhoods), neighborhoods, dados)
    
    f1_on_f2 = calculate_f1(best_f2, dados)
    f2_on_f1 = calculate_f2(best_f1, dados)
    
    extremes = {
        'min_f1': best_f1.f1,
        'max_f1': f1_on_f2,
        'min_f2': best_f2.f2,
        'max_f2': f2_on_f1,
        'sol_f1': best_f1.solution,
        'sol_f2': best_f2.solution
    }
    
    print(f"Extremos calculados: F1 [{extremes['min_f1']:.2f}, {extremes['max_f1']:.2f}], F2 [{extremes['min_f2']:.2f}, {extremes['max_f2']:.2f}]")
    return extremes

def weighted_sum(solution, dados, weight_f1, extremes):
    f1 = calculate_f1(solution, dados)
    f2 = calculate_f2(solution, dados)
    
    f1_norm = (f1 - extremes['min_f1']) / (extremes['max_f1'] - extremes['min_f1'])
    f2_norm = (f2 - extremes['min_f2']) / (extremes['max_f2'] - extremes['min_f2'])
    
    solution.fitness = weight_f1 * f1_norm + (1 - weight_f1) * f2_norm
    solution.f1 = f1
    solution.f2 = f2
    solution.f1_norm = f1_norm
    solution.f2_norm = f2_norm
    return solution.fitness

def calcular_hipervolume_sem_sobreposicao(pontos, ref_point=(1.0, 1.0)):
    """
    Calcula o hipervolume em 2D considerando o ponto de referência (nadir),
    garantindo que áreas sobrepostas não sejam contadas duas vezes.
    
    pontos: lista de tuplas (f1_norm, f2_norm), normalizados e no espaço [0,1].
    ref_point: ponto nadir (geralmente o pior ponto) para cálculo do hipervolume.
    """
    pontos = sorted(pontos, key=lambda x: x[0])
    
    hipervolume = 0.0
    prev_f1 = 0.0
    prev_f2 = ref_point[1]

    for f1, f2 in pontos:
        largura = f1 - prev_f1
        altura = prev_f2 - f2
        if largura > 0 and altura > 0:
            hipervolume += largura * altura
        prev_f1 = f1
        prev_f2 = f2

    largura = ref_point[0] - prev_f1
    altura = prev_f2 - 0.0
    if largura > 0 and altura > 0:
        hipervolume += largura * altura

    return hipervolume

def print_progress(current, total, weight, best_fitness, hv=None):
    progress = (current / total) * 100
    msg = f"Progresso: {current}/{total} ({progress:.1f}%) | Peso: {weight:.3f} | Fitness: {best_fitness:.4f}"
    if hv is not None:
        msg += f" | Hipervolume: {hv:.4f}"
    print(msg)

def generate_pareto_front(dados, extremes, num_points=20, max_iter=1000, num_runs=3):
    weights = np.linspace(0, 1, num_points)
    neighborhoods = [shake_swap, shake_increase, shake_decrease]
    solutions = []
    
    sol_f1 = Solution(extremes['sol_f1'])
    sol_f1.f1 = extremes['min_f1']
    sol_f1.f2 = calculate_f2(sol_f1, dados)
    solutions.append((sol_f1.f1, sol_f1.f2, sol_f1.solution))
    
    sol_f2 = Solution(extremes['sol_f2'])
    sol_f2.f2 = extremes['min_f2']
    sol_f2.f1 = calculate_f1(sol_f2, dados)
    solutions.append((sol_f2.f1, sol_f2.f2, sol_f2.solution))
    
    print("\nIniciando otimização com soma ponderada:")
    print(f"Total de pesos: {len(weights)}")
    print(f"Execuções por peso: {num_runs}")
    print(f"Total de iterações: {len(weights)*num_runs}")
    print("="*60)
    
    all_solutions = []
    
    for i, weight in enumerate(weights):
        best_sol = None
        best_fitness = float('inf')
        
        for run in range(num_runs):
            def ws_obj(solution, d=dados, w=weight, e=extremes):
                return weighted_sum(solution, d, w, e)
            
            best, _ = gvns(lambda: initial_solution(dados), ws_obj, max_iter, 
                          len(neighborhoods), neighborhoods, dados)
            
            if best.fitness < best_fitness:
                best_sol = (best.f1, best.f2, best.solution, best.f1_norm, best.f2_norm)
                best_fitness = best.fitness
            
            # Calcula hipervolume parcial
            current_solutions = solutions + [best_sol[:2]] if best_sol else solutions
            f1_norms = [s[3] for s in all_solutions] + [(s[0]-extremes['min_f1'])/(extremes['max_f1']-extremes['min_f1']) for s in solutions]
            f2_norms = [s[4] for s in all_solutions] + [(s[1]-extremes['min_f2'])/(extremes['max_f2']-extremes['min_f2']) for s in solutions]
            pontos_norm = list(zip(f1_norms, f2_norms))
            hv = calcular_hipervolume_sem_sobreposicao(pontos_norm) if pontos_norm else 0
            
            print_progress(i*num_runs + run + 1, len(weights)*num_runs, weight, best_fitness, hv)
        
        if best_sol:
            solutions.append(best_sol[:3])
            all_solutions.append(best_sol)
    
    print("\nOtimização concluída!")
    print("="*60)
    
    non_dominated = []
    for i, sol_i in enumerate(solutions):
        dominated = False
        for j, sol_j in enumerate(solutions):
            if i != j and (sol_j[0] <= sol_i[0] and sol_j[1] <= sol_i[1]) and (sol_j[0] < sol_i[0] or sol_j[1] < sol_i[1]):
                dominated = True
                break
        if not dominated:
            non_dominated.append(sol_i)
    
    non_dominated.sort(key=lambda x: x[0])
    
    # Calcula hipervolume final
    f1_norms = [(s[0]-extremes['min_f1'])/(extremes['max_f1']-extremes['min_f1']) for s in non_dominated]
    f2_norms = [(s[1]-extremes['min_f2'])/(extremes['max_f2']-extremes['min_f2']) for s in non_dominated]
    pontos_norm = list(zip(f1_norms, f2_norms))
    hv_final = calcular_hipervolume_sem_sobreposicao(pontos_norm)
    
    print(f"\nHipervolume final da fronteira Pareto: {hv_final:.4f}")
    print("="*60)
    
    return non_dominated, hv_final

if __name__ == "__main__":
    dados = load_data()
    extremes = find_extremes(dados)
    pareto_front, hv = generate_pareto_front(dados, extremes, num_points=20, max_iter=1000)
    
    f1_values = [s[0] for s in pareto_front]
    f2_values = [s[1] for s in pareto_front]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(f2_values, f1_values, c='blue', label='Soluções Pareto')
    plt.scatter(extremes['min_f2'], extremes['min_f1'], c='red', marker='*', s=200, label='Solução Ideal F1')
    plt.scatter(extremes['max_f2'], extremes['max_f1'], c='green', marker='*', s=200, label='Solução Ideal F2')
    plt.xlabel('Custo de Falha (F2)')
    plt.ylabel('Custo de Manutenção (F1)')
    plt.title(f'Fronteira Pareto - Hipervolume: {hv:.4f}')
    plt.grid(True)
    plt.legend()
    plt.show()
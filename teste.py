import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from itertools import combinations

# --- Estrutura simples para soluções ---
class Struct:
    pass

def wrap_struct(solution):
    x = Struct()
    x.solution = solution
    return x

# --- Leitura dos dados ---
Equipamentos = pd.read_csv('Dados/EquipDB.csv', header=None, names=["ID", "t0", "cluster", "custo_falha"])
Planos = pd.read_csv('Dados/MPDB.csv', header=None, names=["ID", "k", "custo"]).set_index("ID")
ModeloFalha = pd.read_csv('Dados/ClusterDB.csv', header=None, names=["ID", "eta", "beta"]).set_index("ID")

def manutencao_def(equipamentos, planos):
    dados = Struct()
    dados.equipamentos = equipamentos
    dados.planos = planos
    dados.modelo_falha = ModeloFalha
    dados.n = len(equipamentos)
    dados.custo_por_plano = planos["custo"].to_dict()
    return dados

# --- Solução inicial melhorada ---
def Sol_Inicial(dados, delta_t=5):
    n = dados.n
    t0 = dados.equipamentos['t0'].values
    clusters = dados.equipamentos['cluster'].values
    custo_falha = dados.equipamentos['custo_falha'].values
    eta = dados.modelo_falha.loc[clusters, 'eta'].values
    beta = dados.modelo_falha.loc[clusters, 'beta'].values

    Fi = lambda t, eta, beta: 1 - np.exp(-(t / eta)**beta)
    
    # Calcula probabilidade de falha para cada equipamento sem manutenção
    Fi_t0 = Fi(t0, eta, beta)
    Fi_t0_dt = Fi(t0 + delta_t, eta, beta)
    pi = (Fi_t0_dt - Fi_t0) / (1 - Fi_t0 + 1e-10)
    
    # Ordena equipamentos por custo-benefício (custo de falha * probabilidade)
    custo_beneficio = pi * custo_falha
    indices_ordenados = np.argsort(custo_beneficio)
    
    # Distribui os planos de manutenção de forma mais inteligente
    sol = np.ones(n, dtype=int)
    terc = n // 3
    sol[indices_ordenados[:terc]] = 1  # Menos manutenção para equipamentos menos críticos
    sol[indices_ordenados[terc:2*terc]] = 2
    sol[indices_ordenados[2*terc:]] = 3  # Mais manutenção para equipamentos mais críticos
    
    return sol

# --- Funções objetivo ---
def fobj_f1(x, dados):
    x.fitness = sum(dados.custo_por_plano.get(p, 0) for p in x.solution)
    return x

def fobj_f2(x, dados, delta_t=5):
    t0 = dados.equipamentos['t0'].values
    clusters = dados.equipamentos['cluster'].values
    custo_falha = dados.equipamentos['custo_falha'].values
    planos_k = np.array([dados.planos.loc[p, 'k'] for p in x.solution])

    eta = dados.modelo_falha.loc[clusters, 'eta'].values
    beta = dados.modelo_falha.loc[clusters, 'beta'].values

    Fi = lambda t, eta, beta: 1 - np.exp(-(t / eta)**beta)

    Fi_t0 = Fi(t0, eta, beta)
    Fi_t0_kdt = Fi(t0 + planos_k * delta_t, eta, beta)

    pi = (Fi_t0_kdt - Fi_t0) / (1 - Fi_t0 + 1e-10)
    x.fitness = np.sum(pi * custo_falha)
    return x

# --- Vizinhanças melhoradas ---
def shake_troca(x, r=1, planos=3):
    y = copy.deepcopy(x)
    indices = np.random.choice(len(y.solution), r, replace=False)
    for idx in indices:
        opcoes = [p for p in range(1, planos+1) if p != y.solution[idx]]
        y.solution[idx] = np.random.choice(opcoes)
    return y

def shake_incrementa(x, r=1, planos=3):
    y = copy.deepcopy(x)
    indices = np.random.choice(len(y.solution), r, replace=False)
    for idx in indices:
        if y.solution[idx] < planos:
            y.solution[idx] += 1
    return y

def shake_decrementa(x, r=1):
    y = copy.deepcopy(x)
    indices = np.random.choice(len(y.solution), r, replace=False)
    for idx in indices:
        if y.solution[idx] > 1:
            y.solution[idx] -= 1
    return y

def shake_intermediario(x, r=1, planos=3):
    y = copy.deepcopy(x)
    indices = np.random.choice(len(y.solution), r, replace=False)
    for idx in indices:
        atual = y.solution[idx]
        if atual in [1, planos]:
            y.solution[idx] = 2
        else:
            y.solution[idx] = np.random.randint(1, planos+1)
    return y

# --- Busca local GVNS melhorada ---
def busca_local_gvns(solucao, vizinhancas, fobj, dados, max_avaliacoes, r=1, verbose=False):
    x = copy.deepcopy(solucao)
    x = fobj(x, dados)
    avaliacoes = 0
    melhorou = True
    
    while melhorou and avaliacoes < max_avaliacoes:
        melhorou = False
        for viz in vizinhancas:
            y = viz(x, r)
            y = fobj(y, dados)
            avaliacoes += 1
            
            if y.fitness < x.fitness:
                x = copy.deepcopy(y)
                melhorou = True
                break
                
    return x, avaliacoes

# --- GVNS principal atualizado ---
def gvns(sol_inicial, fobj, max_iter, k_max, vizinhancas, dados, r=10, verbose=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = wrap_struct(sol_inicial())
    x = fobj(x, dados)
    melhor = copy.deepcopy(x)

    avaliacoes = 1
    while avaliacoes < max_iter:
        k = 0
        while k < k_max and avaliacoes < max_iter:
            y = vizinhancas[k](x, r)
            y = fobj(y, dados)
            avaliacoes += 1

            y_local, avals = busca_local_gvns(y, vizinhancas, fobj, dados, max_iter - avaliacoes, r, verbose)
            avaliacoes += avals
            y = y_local

            if y.fitness < melhor.fitness:
                melhor = copy.deepcopy(y)
                x = copy.deepcopy(y)
                k = 0
            else:
                k += 1
    return melhor, avaliacoes

# --- Encontrar extremos com cache ---
def encontrar_extremos(dados, max_iter=4000, verbose=False):
    if hasattr(dados, 'extremos'):
        return dados.extremos

    if verbose:
        print("Calculando extremos monoobjetivo...")

    vizinhancas = [shake_troca, shake_incrementa, shake_decrementa, shake_intermediario]

    # Otimização para F1 (custo de manutenção)
    best_f1, _ = gvns(lambda: Sol_Inicial(dados), fobj_f1, max_iter, len(vizinhancas), vizinhancas, dados, r=10, verbose=False)
    
    # Otimização para F2 (custo de falha)
    best_f2, _ = gvns(lambda: Sol_Inicial(dados), fobj_f2, max_iter, len(vizinhancas), vizinhancas, dados, r=10, verbose=False)

    # Calcula F1 e F2 para as soluções extremas
    f1_on_f2 = fobj_f1(copy.deepcopy(best_f2), dados).fitness
    f2_on_f1 = fobj_f2(copy.deepcopy(best_f1), dados).fitness

    dados.extremos = {
        "min_f1": best_f1.fitness,
        "max_f1": f1_on_f2,
        "min_f2": best_f2.fitness,
        "max_f2": f2_on_f1,
        "sol_f1": best_f1.solution,
        "sol_f2": best_f2.solution
    }
    
    if verbose:
        print(f"Extremos calculados: F1 [{dados.extremos['min_f1']:.2f}, {dados.extremos['max_f1']:.2f}], F2 [{dados.extremos['min_f2']:.2f}, {dados.extremos['max_f2']:.2f}]")

    return dados.extremos

# --- Soma ponderada melhorada ---
def soma_ponderada(x, dados, peso_f1):
    if not hasattr(dados, 'extremos'):
        dados.extremos = encontrar_extremos(dados)

    # Calcular F1 e F2
    f1 = fobj_f1(copy.deepcopy(x), dados).fitness
    f2 = fobj_f2(copy.deepcopy(x), dados).fitness
    
    # Normalizar entre 0 e 1
    min_f1, max_f1 = dados.extremos['min_f1'], dados.extremos['max_f1']
    min_f2, max_f2 = dados.extremos['min_f2'], dados.extremos['max_f2']
    
    f1_norm = (f1 - min_f1) / (max_f1 - min_f1) if max_f1 != min_f1 else 0
    f2_norm = (f2 - min_f2) / (max_f2 - min_f2) if max_f2 != min_f2 else 0
    
    # Combinação linear com pesos entre 0 e 1
    peso_f1_normalizado = peso_f1  # Já espera pesos entre 0 e 1 agora
    x.fitness = peso_f1_normalizado * f1_norm + (1 - peso_f1_normalizado) * f2_norm
    x.f1 = f1
    x.f2 = f2
    x.f1_norm = f1_norm
    x.f2_norm = f2_norm
    
    return x

# --- Funções de dominância e filtragem melhoradas ---
def is_dominated(p, q):
    """Retorna True se p é dominado por q (p e q são tuplas (f1, f2))"""
    return (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])

def filtrar_nao_dominadas(solucoes, max_pontos=50):
    """Filtra soluções não dominadas e limita o número máximo de pontos"""
    # Converte para lista de tuplas (f1, f2, solução)
    pontos = [(s[0], s[1], s[2]) for s in solucoes]
    
    # Filtra não-dominadas
    nao_dominadas = []
    for i, p in enumerate(pontos):
        dominado = False
        for j, q in enumerate(pontos):
            if i != j and is_dominated(p[:2], q[:2]):
                dominado = True
                break
        if not dominado:
            nao_dominadas.append(p)
    
    # Ordena por F1
    nao_dominadas.sort(key=lambda x: x[0])
    
    # Limita o número de pontos se necessário
    if len(nao_dominadas) > max_pontos:
        # Seleciona pontos igualmente espaçados
        indices = np.linspace(0, len(nao_dominadas)-1, max_pontos, dtype=int)
        nao_dominadas = [nao_dominadas[i] for i in indices]
    
    return nao_dominadas

# --- Geração de pesos melhorada ---
def gerar_pesos(num_pontos=20):
    """Gera pesos igualmente espaçados entre 0 e 1"""
    return np.linspace(0, 1, num_pontos)

# --- Refinamento da fronteira Pareto ---
def refinamento_fronteira(fronteira, dados, max_iter=100):
    """Refina as soluções na fronteira Pareto com busca local"""
    vizinhancas = [shake_troca, shake_decrementa, shake_incrementa, shake_intermediario]
    fronteira_refinada = []
    
    for f1, f2, sol in fronteira:
        # Cria função objetivo personalizada para manter o trade-off
        peso = (f1 - dados.extremos['min_f1']) / (dados.extremos['max_f1'] - dados.extremos['min_f1'])
        peso = np.clip(peso, 0, 1)
        
        def fobj_refinamento(x, d=dados, p=peso):
            return soma_ponderada(x, d, p)
        
        # Aplica busca local
        x = wrap_struct(sol)
        x_refinado, _ = busca_local_gvns(x, vizinhancas, fobj_refinamento, dados, max_iter, r=5)
        
        # Atualiza valores F1 e F2
        f1_ref = fobj_f1(x_refinado, dados).fitness
        f2_ref = fobj_f2(x_refinado, dados).fitness
        fronteira_refinada.append((f1_ref, f2_ref, x_refinado.solution))
    
    return filtrar_nao_dominadas(fronteira_refinada)

# --- Geração da fronteira Pareto com soma ponderada ---
def gerar_fronteira_pareto(dados, num_pontos=20, max_iter=1000, num_repeticoes=3):
    """Gera a fronteira Pareto usando soma ponderada com múltiplas execuções e refinamento"""
    pesos = gerar_pesos(num_pontos)
    vizinhancas = [shake_troca, shake_decrementa, shake_incrementa, shake_intermediario]
    solucoes = []
    
    # Garante que temos os extremos
    if not hasattr(dados, 'extremos'):
        encontrar_extremos(dados)
    
    # Adiciona soluções extremas
    solucoes.append((dados.extremos['min_f1'], dados.extremos['max_f2'], dados.extremos['sol_f1']))
    solucoes.append((dados.extremos['max_f1'], dados.extremos['min_f2'], dados.extremos['sol_f2']))
    
    # Otimização para cada peso
    for peso in pesos:
        melhor_sol = None
        melhor_fit = float('inf')
        
        for _ in range(num_repeticoes):
            fobj = lambda x, d=dados, p=peso: soma_ponderada(x, d, p)
            best, _ = gvns(lambda: Sol_Inicial(dados), fobj, max_iter, len(vizinhancas), vizinhancas, dados, r=5, verbose=False)
            
            f1 = fobj_f1(best, dados).fitness
            f2 = fobj_f2(best, dados).fitness
            fit = fobj(best, dados).fitness
            
            if fit < melhor_fit:
                melhor_fit = fit
                melhor_sol = (f1, f2, best.solution)
        
        if melhor_sol:
            solucoes.append(melhor_sol)
    
    # Filtra não-dominadas
    fronteira = filtrar_nao_dominadas(solucoes)
    
    # Aplica refinamento
    fronteira_refinada = refinamento_fronteira(fronteira, dados, max_iter//2)
    
    return [s[0] for s in fronteira_refinada], [s[1] for s in fronteira_refinada], [s[2] for s in fronteira_refinada]

# --- Cálculo de hipervolume melhorado ---
def calcular_hipervolume(pontos, ref_point=(1.0, 1.0)):
    """Calcula o hipervolume para pontos normalizados"""
    if not pontos:
        return 0.0
    
    # Ordena pontos por F1
    pontos = sorted(pontos, key=lambda x: x[0])
    
    hv = 0.0
    prev_f1, prev_f2 = 0.0, ref_point[1]
    
    for f1, f2 in pontos:
        largura = f1 - prev_f1
        altura = prev_f2 - f2
        hv += largura * altura
        prev_f1, prev_f2 = f1, f2
    
    # Adiciona área final
    largura = ref_point[0] - prev_f1
    altura = prev_f2 - 0.0
    hv += largura * altura
    
    return hv

# --- Execução múltipla com análise ---
def executar_multiplas_execucoes(dados, tecnica, num_exec=5, num_pontos=20, max_iter=1000):
    """Executa o algoritmo várias vezes e coleta resultados"""
    resultados = []
    hipervolumes = []
    
    for i in range(num_exec):
        print(f"Execução {i+1}/{num_exec} para {tecnica}")
        
        if tecnica == 'soma_ponderada':
            f1, f2, _ = gerar_fronteira_pareto(dados, num_pontos, max_iter)
        else:
            raise ValueError("Técnica desconhecida")
        
        resultados.append((f1, f2))
        
        # Calcula hipervolume
        f1_norm, f2_norm = normalizar_solucoes(f1, f2, dados.extremos)
        hv = calcular_hipervolume(list(zip(f1_norm, f2_norm)))
        hipervolumes.append(hv)
        print(f"Hipervolume da execução {i+1}: {hv:.4f}")
    
    return resultados, hipervolumes

def normalizar_solucoes(f1_vals, f2_vals, extremos):
    """Normaliza os valores entre 0 e 1 usando os extremos"""
    min_f1, max_f1 = extremos['min_f1'], extremos['max_f1']
    min_f2, max_f2 = extremos['min_f2'], extremos['max_f2']
    
    f1_norm = (np.array(f1_vals) - min_f1) / (max_f1 - min_f1)
    f2_norm = (np.array(f2_vals) - min_f2) / (max_f2 - min_f2)
    
    return f1_norm, f2_norm

# --- Visualização melhorada ---
def plotar_fronteiras_multiplas(lista_fronteiras, extremos, tecnica='soma_ponderada'):
    """Plota as fronteiras de múltiplas execuções"""
    plt.figure(figsize=(12, 8))
    cores = plt.cm.viridis(np.linspace(0, 1, len(lista_fronteiras)))
    
    for i, (f1, f2) in enumerate(lista_fronteiras):
        # Plota pontos não normalizados
        plt.scatter(f2, f1, s=40, color=cores[i], alpha=0.7, label=f'Execução {i+1}')
        plt.plot(f2, f1, color=cores[i], alpha=0.3)
    
    # Adiciona pontos extremos
    plt.scatter(extremos['min_f2'], extremos['min_f1'], s=100, c='red', marker='*', label='Solução Ideal F1')
    plt.scatter(extremos['max_f2'], extremos['max_f1'], s=100, c='blue', marker='*', label='Solução Ideal F2')
    
    plt.title(f'Fronteiras Não Dominadas - Técnica: {tecnica}')
    plt.xlabel("Custo de Falha (F2)")
    plt.ylabel("Custo de Manutenção (F1)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

# --- Execução principal ---
if __name__ == "__main__":
    # Carrega e prepara os dados
    dados = manutencao_def(Equipamentos, Planos)
    
    # Encontra soluções extremas
    dados.extremos = encontrar_extremos(dados, max_iter=1000, verbose=True)
    
    # Executa a soma ponderada múltiplas vezes
    tecnicas = ['soma_ponderada']
    execs, hvs = executar_multiplas_execucoes(dados, tecnicas[0], num_exec=5, num_pontos=20, max_iter=1000)
    
    # Exibe estatísticas do hipervolume
    print("\nEstatísticas do Hipervolume:")
    print(f"Média: {np.mean(hvs):.4f}")
    print(f"Desvio Padrão: {np.std(hvs):.4f}")
    print(f"Mínimo: {np.min(hvs):.4f}")
    print(f"Máximo: {np.max(hvs):.4f}")
    
    # Plota os resultados
    plotar_fronteiras_multiplas(execs, dados.extremos, tecnicas[0])
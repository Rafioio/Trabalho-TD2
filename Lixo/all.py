import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
from scipy.stats import beta

# Estrutura auxiliar
class Struct:
    pass

def wrap_struct(solution_array):
    x = Struct()
    x.solution = np.array(solution_array) # Garante que é um array numpy
    x.fitness = None # Inicializa o fitness
    return x

# Leitura dos arquivos e definição dos dados
def manutencao_def(equipamentos_df, planos_df, modelo_falha_df):
    dados = Struct()
    dados.equipamentos = equipamentos_df
    dados.planos = planos_df.set_index("ID")
    dados.modelo_falha = modelo_falha_df.set_index("ID")
    dados.n = len(equipamentos_df)
    dados.custo_por_plano = dados.planos["custo"].to_dict()
    dados.extremos = None # Inicializa extremos
    return dados

# Geração da Solução Inicial
def Sol_Inicial(dados, delta_t=5):
    n = dados.n
    try:
        custo_manut_base = dados.planos.loc[1, 'custo']
        if custo_manut_base == 0:
            custo_manut_base = 1e-10
    except KeyError:
        print("Aviso: Plano 1 não encontrado ou sem custo em dados.planos. Usando custo base de 1e-10.")
        custo_manut_base = 1e-10


    t0 = dados.equipamentos['t0'].values
    clusters = dados.equipamentos['cluster'].values
    custo_falha = dados.equipamentos['custo de falha'].values

    if not all(c in dados.modelo_falha.index for c in clusters):
        raise ValueError("Um ou mais clusters de equipamentos não encontrados em ModeloFalha.")

    eta = dados.modelo_falha.loc[clusters, 'eta'].values
    beta_params = dados.modelo_falha.loc[clusters, 'beta'].values

    def Fi(t, eta_vals, beta_p_vals):
        t_safe = np.maximum(t, 1e-9)
        eta_safe = np.maximum(eta_vals, 1e-9)
        ratio = t_safe / eta_safe
        exponent = np.clip(ratio ** beta_p_vals, 0, 700)
        return 1 - np.exp(-exponent)


    Fi_t0 = Fi(t0, eta, beta_params)
    Fi_t0_dt = Fi(t0 + delta_t, eta, beta_params)

    pi = (Fi_t0_dt - Fi_t0) / (1 - Fi_t0 + 1e-10)
    custo_beneficio = (pi * custo_falha) / custo_manut_base
    indices_ordenados = np.argsort(custo_beneficio)[::-1]

    sol_inicial_array = np.ones(n, dtype=int)
    terco = n // 3
    for i, idx in enumerate(indices_ordenados):
        if i < terco:
            sol_inicial_array[idx] = 1
        elif i < 2 * terco:
            sol_inicial_array[idx] = 2
        else:
            sol_inicial_array[idx] = 3
    return sol_inicial_array


# Funções Objetivo
def fobj_f1(x_struct, dados):
    planos_sol = x_struct.solution
    custo_total = 0
    for plano_id in planos_sol:
        try:
            custo_total += dados.custo_por_plano[plano_id]
        except KeyError:
            custo_total += 0
    x_struct.fitness = custo_total
    return x_struct

def fobj_f2(x_struct, dados, delta_t=5):
    planos_sol = x_struct.solution
    equipamentos = dados.equipamentos
    modelo = dados.modelo_falha
    planos_df = dados.planos

    t0 = equipamentos['t0'].values
    clusters = equipamentos['cluster'].values
    custo_falha = equipamentos['custo de falha'].values

    eta = modelo.loc[clusters, 'eta'].values
    beta_params = modelo.loc[clusters, 'beta'].values

    k_array = np.zeros(len(planos_sol))
    for i, plano_id in enumerate(planos_sol):
        try:
            k_array[i] = planos_df.loc[plano_id, 'k']
        except KeyError:
            k_array[i] = 1

    def Fi(t, eta_vals, beta_p_vals):
        t_safe = np.maximum(t, 1e-9)
        eta_safe = np.maximum(eta_vals, 1e-9)
        ratio = t_safe / eta_safe
        exponent = np.clip(ratio ** beta_p_vals, 0, 700)
        return 1 - np.exp(-exponent)

    Fi_t0 = Fi(t0, eta, beta_params)
    Fi_t0_kdt = Fi(t0 + k_array * delta_t, eta, beta_params)

    pi = (Fi_t0_kdt - Fi_t0) / (1 - Fi_t0 + 1e-10)
    custo = np.sum(pi * custo_falha)
    x_struct.fitness = custo
    return x_struct


# Busca Local e Estruturas de Vizinhança (Shake)
def busca_local_gvns(solucao_struct, vizinhancas_list, fobj_func, dados_obj, max_avaliacoes_bl, r_shake, verbose=False):
    x_atual = copy.deepcopy(solucao_struct)
    avaliacoes_feitas = 0
    melhoria_encontrada = True

    while melhoria_encontrada and avaliacoes_feitas < max_avaliacoes_bl:
        melhoria_encontrada = False
        for viz_func in vizinhancas_list:
            y_vizinho_struct = viz_func(x_atual, r_shake)
            y_vizinho_struct = fobj_func(y_vizinho_struct, dados_obj)
            avaliacoes_feitas += 1
            if verbose:
                print(f"BL Avaliação {avaliacoes_feitas}: fitness atual {x_atual.fitness:.4f}, vizinho {y_vizinho_struct.fitness:.4f}")
            if y_vizinho_struct.fitness < x_atual.fitness:
                x_atual = copy.deepcopy(y_vizinho_struct)
                melhoria_encontrada = True
                if verbose:
                    print(f"BL Melhora encontrada: fitness {x_atual.fitness:.4f}")
                break
        if avaliacoes_feitas >= max_avaliacoes_bl:
            break
    return x_atual, avaliacoes_feitas

def shake_troca(x_struct, r, num_planos_total=3):
    y_struct = copy.deepcopy(x_struct)
    n_equip = len(y_struct.solution)
    indices_modificar = np.random.choice(n_equip, size=min(r, n_equip), replace=False)
    for idx in indices_modificar:
        plano_atual = y_struct.solution[idx]
        opcoes_disponiveis = [p for p in range(1, num_planos_total + 1) if p != plano_atual]
        if opcoes_disponiveis:
            y_struct.solution[idx] = np.random.choice(opcoes_disponiveis)
    return y_struct

def shake_incrementa(x_struct, r, num_planos_total=3):
    y_struct = copy.deepcopy(x_struct)
    n_equip = len(y_struct.solution)
    indices_modificar = np.random.choice(n_equip, size=min(r, n_equip), replace=False)
    for idx in indices_modificar:
        if y_struct.solution[idx] < num_planos_total:
            y_struct.solution[idx] += 1
    return y_struct

def shake_decrementa(x_struct, r):
    y_struct = copy.deepcopy(x_struct)
    n_equip = len(y_struct.solution)
    indices_modificar = np.random.choice(n_equip, size=min(r, n_equip), replace=False)
    for idx in indices_modificar:
        if y_struct.solution[idx] > 1:
            y_struct.solution[idx] -= 1
    return y_struct

def shake_intermediario(x_struct, r, num_planos_total=3):
    y_struct = copy.deepcopy(x_struct)
    n_equip = len(y_struct.solution)
    indices_modificar = np.random.choice(n_equip, size=min(r, n_equip), replace=False)
    plano_intermediario = 2
    for idx in indices_modificar:
        plano_atual = y_struct.solution[idx]
        if plano_atual == 1 or plano_atual == num_planos_total:
            if plano_intermediario != plano_atual:
                 y_struct.solution[idx] = plano_intermediario
        else:
            opcoes_disponiveis = [p for p in range(1, num_planos_total + 1) if p != plano_atual]
            if opcoes_disponiveis:
                y_struct.solution[idx] = np.random.choice(opcoes_disponiveis)
    return y_struct


# GVNS
def gvns_base(sol_inicial_func, fobj_calc_func, max_iter_gvns, k_max_gvns, vizinhancas_gvns, dados_obj, r_shake_gvns=10, verbose_gvns=False, seed_gvns=None, usar_prioridade_extremos_gvns=False):
    if seed_gvns is not None:
        np.random.seed(seed_gvns)

    x_corrente_struct = wrap_struct(sol_inicial_func())
    x_corrente_struct = fobj_calc_func(x_corrente_struct, dados_obj)
    melhor_sol_struct = copy.deepcopy(x_corrente_struct)
    num_avaliacoes_total = 1

    if usar_prioridade_extremos_gvns:
        if not hasattr(dados_obj, 'extremos') or dados_obj.extremos is None:
            print("Aviso (GVNS): dados.extremos não encontrado ou é None. Calculando agora.")
            encontrar_extremos(dados_obj, verbose_extremos=False)
        if not hasattr(dados_obj, 'extremos') or dados_obj.extremos is None:
             raise AttributeError("GVNS: dados.extremos não pôde ser calculado.")
        extremos = dados_obj.extremos
        min_f1_ext, max_f1_ext = extremos['min_f1'], extremos['max_f1']
    else:
        min_f1_ext = max_f1_ext = None

    iter_sem_melhora_global = 0

    while num_avaliacoes_total < max_iter_gvns:
        k_shake = 0
        while k_shake < k_max_gvns and num_avaliacoes_total < max_iter_gvns:
            viz_ordem_atual = vizinhancas_gvns
            if usar_prioridade_extremos_gvns and min_f1_ext is not None:
                f1_da_melhor_sol = fobj_f1(copy.deepcopy(melhor_sol_struct), dados_obj).fitness
                diff_min = abs(f1_da_melhor_sol - min_f1_ext)
                diff_max = abs(f1_da_melhor_sol - max_f1_ext)
                range_f1 = max_f1_ext - min_f1_ext if max_f1_ext != min_f1_ext else 1.0

                if diff_min < range_f1 * 0.1:
                    viz_ordem_atual = [shake_incrementa, shake_troca, shake_decrementa, shake_intermediario]
                elif diff_max < range_f1 * 0.1:
                    viz_ordem_atual = [shake_decrementa, shake_troca, shake_incrementa, shake_intermediario]

            idx_viz_shake = k_shake % len(viz_ordem_atual)
            x_perturbado_struct = viz_ordem_atual[idx_viz_shake](x_corrente_struct, r_shake_gvns)
            x_perturbado_struct = fobj_calc_func(x_perturbado_struct, dados_obj)
            num_avaliacoes_total +=1

            x_local_opt_struct, avals_bl = busca_local_gvns(x_perturbado_struct, viz_ordem_atual, fobj_calc_func, dados_obj, max_iter_gvns - num_avaliacoes_total, r_shake_gvns, verbose=verbose_gvns)
            num_avaliacoes_total += avals_bl

            if x_local_opt_struct.fitness < melhor_sol_struct.fitness:
                if verbose_gvns:
                    print(f"GVNS: Nova melhor solução global encontrada: {x_local_opt_struct.fitness:.4f} (anterior: {melhor_sol_struct.fitness:.4f})")
                melhor_sol_struct = copy.deepcopy(x_local_opt_struct)
                x_corrente_struct = copy.deepcopy(x_local_opt_struct)
                k_shake = 0
                iter_sem_melhora_global = 0
            else:
                x_corrente_struct = copy.deepcopy(x_local_opt_struct)
                k_shake += 1
                iter_sem_melhora_global += 1
            
            if num_avaliacoes_total >= max_iter_gvns: break
        if iter_sem_melhora_global > k_max_gvns * 5 and verbose_gvns:
            pass

    return melhor_sol_struct, None


# Cálculo dos Extremos
def encontrar_extremos(dados_obj, max_iter_extremos=2000, verbose_extremos=True, seed_extremos=None):
    if hasattr(dados_obj, 'extremos') and dados_obj.extremos is not None:
        return dados_obj.extremos
    if verbose_extremos:
        print(" Calculando extremos monoobjetivo...")

    vizinhancas_padrao = [shake_troca, shake_incrementa, shake_decrementa, shake_intermediario]
    r_padrao = max(5, int(len(dados_obj.equipamentos) * 0.05))

    def otimizar_mono(fobj_mono_func):
        sol_otima, _ = gvns_base(
            sol_inicial_func=lambda: Sol_Inicial(dados_obj),
            fobj_calc_func=fobj_mono_func,
            max_iter_gvns=max_iter_extremos,
            k_max_gvns=len(vizinhancas_padrao),
            vizinhancas_gvns=vizinhancas_padrao,
            dados_obj=dados_obj,
            r_shake_gvns=r_padrao,
            seed_gvns=seed_extremos,
            verbose_gvns=False,
            usar_prioridade_extremos_gvns=False
        )
        return sol_otima

    best_s_f1_struct = otimizar_mono(fobj_f1)
    best_s_f2_struct = otimizar_mono(fobj_f2)

    min_f1_val = best_s_f1_struct.fitness
    f2_at_min_f1 = fobj_f2(copy.deepcopy(best_s_f1_struct), dados_obj).fitness

    min_f2_val = best_s_f2_struct.fitness
    f1_at_min_f2 = fobj_f1(copy.deepcopy(best_s_f2_struct), dados_obj).fitness

    dados_obj.extremos = {
        "min_f1": min_f1_val, "max_f1": f1_at_min_f2,
        "min_f2": min_f2_val, "max_f2": f2_at_min_f1
    }
    if verbose_extremos:
        print(f"Extremos calculados:") # Removido emoji para garantir
        # CORREÇÃO APLICADA AQUI:
        print(f" - F1: {dados_obj.extremos['min_f1']:.2f} (F2={f2_at_min_f1:.2f}) -> {dados_obj.extremos['max_f1']:.2f} (F2={min_f2_val:.2f} quando F1 é otimizado por F2)")
        print(f" - F2: {dados_obj.extremos['min_f2']:.2f} (F1={f1_at_min_f2:.2f}) -> {dados_obj.extremos['max_f2']:.2f} (F1={min_f1_val:.2f} quando F2 é otimizado por F1)")
    return dados_obj.extremos


# Soma Ponderada
def soma_ponderada_fobj(x_struct, dados_obj, peso_f1_percent):
    if not hasattr(dados_obj, 'extremos') or dados_obj.extremos is None:
        encontrar_extremos(dados_obj, verbose_extremos=False)
    if not hasattr(dados_obj, 'extremos') or dados_obj.extremos is None:
        raise ValueError("Extremos não puderam ser calculados para a soma ponderada.")

    f1_val = fobj_f1(copy.deepcopy(x_struct), dados_obj).fitness
    f2_val = fobj_f2(copy.deepcopy(x_struct), dados_obj).fitness

    ext = dados_obj.extremos
    min_f1, max_f1 = ext['min_f1'], ext['max_f1']
    min_f2, max_f2 = ext['min_f2'], ext['max_f2']

    range_f1 = max_f1 - min_f1 if (max_f1 - min_f1) != 0 else 1.0
    range_f2 = max_f2 - min_f2 if (max_f2 - min_f2) != 0 else 1.0

    f1_norm = (f1_val - min_f1) / range_f1
    f2_norm = (f2_val - min_f2) / range_f2
    
    f1_norm = np.clip(f1_norm, 0, 1)
    f2_norm = np.clip(f2_norm, 0, 1)

    x_struct.fitness = (peso_f1_percent / 100.0 * f1_norm + (100 - peso_f1_percent) / 100.0 * f2_norm)
    return x_struct

def gerar_pesos_beta(num_pontos_pesos=50, a_beta=2, b_beta=2, escala_pesos=100):
    if num_pontos_pesos <= 0: return np.array([])
    if num_pontos_pesos == 1: probs = np.array([0.5])
    else: probs = np.linspace(1e-6, 1.0 - 1e-6, num_pontos_pesos)
    
    beta_vals = beta.ppf(probs, a_beta, b_beta)
    pesos_f1 = np.sort(beta_vals * escala_pesos)
    return pesos_f1

# Dominação e Filtragem
def is_dominated(p_obj_vals, q_obj_vals):
    return all(q_val <= p_val for q_val, p_val in zip(q_obj_vals, p_obj_vals)) and \
           any(q_val < p_val for q_val, p_val in zip(q_obj_vals, p_obj_vals))

def filtrar_nao_dominadas(solucoes_list_tuples, max_pontos_filtro=50):
    if not solucoes_list_tuples: return []
    
    nao_dominadas_indices = []
    for i in range(len(solucoes_list_tuples)):
        dominado_flag = False
        for j in range(len(solucoes_list_tuples)):
            if i == j: continue
            if is_dominated(solucoes_list_tuples[i][:2], solucoes_list_tuples[j][:2]):
                dominado_flag = True
                break
        if not dominado_flag:
            nao_dominadas_indices.append(i)
    
    nao_dominadas_filtradas = [solucoes_list_tuples[i] for i in nao_dominadas_indices]
    nao_dominadas_filtradas = sorted(nao_dominadas_filtradas, key=lambda s: s[0])

    if len(nao_dominadas_filtradas) > max_pontos_filtro:
        indices_selecionados = np.linspace(0, len(nao_dominadas_filtradas) - 1, max_pontos_filtro, dtype=int)
        nao_dominadas_filtradas = [nao_dominadas_filtradas[i] for i in indices_selecionados]
    return nao_dominadas_filtradas

# Geração da Fronteira de Pareto (Soma Ponderada)
def gerar_fronteira_pareto_sp(dados_obj, num_pesos=30, max_iter_sp=1000, num_repet_sp=3, peso_func_sp=None):
    if peso_func_sp is None:
        peso_func_sp = lambda n: gerar_pesos_beta(n, a_beta=2, b_beta=2)
    
    pesos_gerados = peso_func_sp(num_pesos)
    vizinhancas_sp = [shake_troca, shake_decrementa, shake_incrementa, shake_intermediario]
    todas_solucoes_encontradas = []
    r_dinamico = max(5, int(len(dados_obj.equipamentos) * 0.1))

    if not hasattr(dados_obj, 'extremos') or dados_obj.extremos is None:
        print("Calculando extremos para normalização na Soma Ponderada (gerar_fronteira_pareto_sp)...")
        encontrar_extremos(dados_obj, verbose_extremos=True)

    for i, peso_f1_atual in enumerate(pesos_gerados):
        print(f"Soma Ponderada - Processando peso {i+1}/{len(pesos_gerados)}: F1={peso_f1_atual:.1f}%")
        melhor_sol_para_peso_tupla = None
        melhor_fitness_para_peso_val = float('inf')

        for rep in range(num_repet_sp):
            fobj_sp_atual = lambda x_s, d_o=dados_obj, p=peso_f1_atual: soma_ponderada_fobj(x_s, d_o, p)
            
            best_sol_struct, _ = gvns_base(
                sol_inicial_func=lambda: Sol_Inicial(dados_obj),
                fobj_calc_func=fobj_sp_atual,
                max_iter_gvns=max_iter_sp,
                k_max_gvns=len(vizinhancas_sp),
                vizinhancas_gvns=vizinhancas_sp,
                dados_obj=dados_obj,
                r_shake_gvns=r_dinamico,
                usar_prioridade_extremos_gvns=False
            )
            
            if best_sol_struct.fitness < melhor_fitness_para_peso_val:
                melhor_fitness_para_peso_val = best_sol_struct.fitness
                f1_val_atual = fobj_f1(copy.deepcopy(best_sol_struct), dados_obj).fitness
                f2_val_atual = fobj_f2(copy.deepcopy(best_sol_struct), dados_obj).fitness
                melhor_sol_para_peso_tupla = (f1_val_atual, f2_val_atual, copy.deepcopy(best_sol_struct.solution))
        
        if melhor_sol_para_peso_tupla is not None:
            todas_solucoes_encontradas.append(melhor_sol_para_peso_tupla)

    fronteira_final = filtrar_nao_dominadas(todas_solucoes_encontradas, max_pontos_filtro=num_pesos * 2)
    f1_resultados = [s[0] for s in fronteira_final]
    f2_resultados = [s[1] for s in fronteira_final]
    solucoes_arrays = [s[2] for s in fronteira_final]
    return f1_resultados, f2_resultados, solucoes_arrays


# Epsilon-Restrito
def fobj_epsilon_restrito(x_struct, dados_obj, epsilon_val, objetivo_principal_str='f1'):
    f1_val = fobj_f1(copy.deepcopy(x_struct), dados_obj).fitness
    f2_val = fobj_f2(copy.deepcopy(x_struct), dados_obj).fitness
    penalidade_grande = 1e9

    if objetivo_principal_str == 'f1':
        if f2_val <= epsilon_val + 1e-9:
            x_struct.fitness = f1_val
        else:
            x_struct.fitness = f1_val + penalidade_grande * (f2_val - epsilon_val)
    else:
        if f1_val <= epsilon_val + 1e-9:
            x_struct.fitness = f2_val
        else:
            x_struct.fitness = f2_val + penalidade_grande * (f1_val - epsilon_val)
    return x_struct

def gerar_fronteira_epsilon(dados_obj, num_pontos_eps=20, max_iter_eps=1000, obj_principal_eps='f1'):
    print(f"Calculando extremos para determinar faixa de epsilon (obj. principal: {obj_principal_eps})...")
    if not hasattr(dados_obj, 'extremos') or dados_obj.extremos is None:
        encontrar_extremos(dados_obj, verbose_extremos=True)
    ext = dados_obj.extremos

    if obj_principal_eps == 'f1': eps_min, eps_max = ext['min_f2'], ext['max_f2']
    else: eps_min, eps_max = ext['min_f1'], ext['max_f1']

    if eps_min == eps_max:
        epsilons_gerados = np.array([eps_min]) if num_pontos_eps > 0 else np.array([])
    elif num_pontos_eps == 1: epsilons_gerados = np.array([(eps_min + eps_max) / 2])
    else: epsilons_gerados = np.linspace(eps_min, eps_max, num_pontos_eps)

    f1_res, f2_res, sols_res = [], [], []
    vizinhancas_eps = [shake_troca, shake_incrementa, shake_decrementa, shake_intermediario]
    r_eps = max(5, int(len(dados_obj.equipamentos) * 0.05))

    for i, eps_atual in enumerate(epsilons_gerados):
        print(f"\nEpsilon-Restrito ({obj_principal_eps}) - Otimizando com epsilon = {eps_atual:.2f} (Progresso: {i+1}/{len(epsilons_gerados)})")
        fobj_eps_atual = lambda x_s, d_o=dados_obj, e=eps_atual: fobj_epsilon_restrito(x_s, d_o, e, obj_principal_eps)
        
        best_s_eps, _ = gvns_base(
            sol_inicial_func=lambda: Sol_Inicial(dados_obj),
            fobj_calc_func=fobj_eps_atual,
            max_iter_gvns=max_iter_eps,
            k_max_gvns=len(vizinhancas_eps),
            vizinhancas_gvns=vizinhancas_eps,
            dados_obj=dados_obj,
            r_shake_gvns=r_eps,
            usar_prioridade_extremos_gvns=True
        )
        f1_obtido = fobj_f1(copy.deepcopy(best_s_eps), dados_obj).fitness
        f2_obtido = fobj_f2(copy.deepcopy(best_s_eps), dados_obj).fitness
        valido = (obj_principal_eps == 'f1' and f2_obtido <= eps_atual + 1e-9) or \
                 (obj_principal_eps == 'f2' and f1_obtido <= eps_atual + 1e-9)
        
        print(f"Epsilon={eps_atual:.2f}, F1 obtido={f1_obtido:.2f}, F2 obtido={f2_obtido:.2f}, Válido={'Sim' if valido else 'Não'}")
        if valido:
            f1_res.append(f1_obtido)
            f2_res.append(f2_obtido)
            sols_res.append(copy.deepcopy(best_s_eps.solution))

    if f1_res:
        fronteira_eps_tuplas = filtrar_nao_dominadas(list(zip(f1_res, f2_res, sols_res)), max_pontos_filtro=len(f1_res))
        return [s[0] for s in fronteira_eps_tuplas], [s[1] for s in fronteira_eps_tuplas], [s[2] for s in fronteira_eps_tuplas]
    return [], [], []


# Plotagem
def plotar_resultados_comparativos(dados_obj, num_pontos_pareto_plot=30, max_iter_plot=1000):
    print("\n--- Gerando fronteira com Soma Ponderada ---")
    f1_pw, f2_pw, _ = gerar_fronteira_pareto_sp(dados_obj, num_pesos=num_pontos_pareto_plot, max_iter_sp=max_iter_plot)
    
    plt.figure(figsize=(14, 9))
    if f1_pw and f2_pw:
        plt.scatter(f2_pw, f1_pw, c='blue', marker='o', s=70, label='Soma Ponderada', alpha=0.9, edgecolors='k', linewidth=0.5)
    else:
        print("Nenhum ponto da Soma Ponderada para plotar.")
    
    plt.title("Comparação de Abordagens Multiobjetivo", fontsize=16)
    plt.xlabel("Custo de Falha (F2)", fontsize=14)
    plt.ylabel("Custo de Manutenção (F1)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    all_f1, all_f2 = [], []
    if f1_pw: all_f1.extend(f1_pw)
    if f2_pw: all_f2.extend(f2_pw)

    if all_f1 and all_f2:
        f1_min_plot, f1_max_plot = min(all_f1), max(all_f1)
        f2_min_plot, f2_max_plot = min(all_f2), max(all_f2)
        f1_range = f1_max_plot - f1_min_plot if f1_max_plot > f1_min_plot else 10
        f2_range = f2_max_plot - f2_min_plot if f2_max_plot > f2_min_plot else 10
        
        plt.xlim(f2_min_plot - 0.05 * f2_range, f2_max_plot + 0.05 * f2_range)
        plt.ylim(f1_min_plot - 0.05 * f1_range, f1_max_plot + 0.05 * f1_range)
    else:
        plt.xlim(800, 2000)
        plt.ylim(0, 1200)
        
    plt.tight_layout()
    plt.show()

# Execução Principal
if __name__ == "__main__":
    path_equip_csv = 'Dados/EquipDB.csv'
    path_planos_csv = 'Dados/MPDB.csv'
    path_cluster_csv = 'Dados/ClusterDB.csv'

    try:
        Equipamentos_df = pd.read_csv(path_equip_csv, header=None, names=["ID", "t0", "cluster", "custo de falha"])
        Planos_df = pd.read_csv(path_planos_csv, header=None, names=["ID", "k", "custo"])
        ModeloFalha_df = pd.read_csv(path_cluster_csv, header=None, names=["ID", "eta", "beta"])
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique os caminhos: {e}")
        exit()

    dados_globais = manutencao_def(Equipamentos_df, Planos_df, ModeloFalha_df)
    
    num_pontos_fronteira = 10
    max_iteracoes_gvns = 1000

    plotar_resultados_comparativos(dados_globais, 
                                   num_pontos_pareto_plot=num_pontos_fronteira, 
                                   max_iter_plot=max_iteracoes_gvns)

    print("\n--- Execução Concluída ---")
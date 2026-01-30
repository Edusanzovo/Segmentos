import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# CONFIGURAÇÃO CENTRALIZADA

BAR_FILE_PATH = 'WDO.csv' 
SEGMENTATION_PATTERN_ID = 2003 
MAX_PULLBACK_N = 2

PARAM_MATRIX = {

    'LATERAL': {
        'L_FORCE': 1.8,
        'P_REVERSAL': 0.20,
        'THRESHOLD_PCT': 0.55,
        'RAL_MAX_N': 3,          
        'CONV_THRESH': {
            'FORTE': 1.0,
            'MEDIA': 1.0,
            'FRACA': 1.0
        }
    },

    'ASCENDENTE_FRACA': {
        'L_FORCE': 2.0,
        'P_REVERSAL': 0.20,      
        'THRESHOLD_PCT': 0.55,
        'CONV_THRESH': {
            'FORTE': 1.0,
            'MEDIA': 0.6,
            'FRACA': 0.10
        }
    },

    'ASCENDENTE_MEDIA': {
        'L_FORCE': 4.5,
        'P_REVERSAL': 0.15,     
        'THRESHOLD_PCT': 0.50,
        'CONV_THRESH': {
            'FORTE': 1.0,
            'MEDIA': 0.6,
            'FRACA': 0.15
        }
    },

    'ASCENDENTE_FORTE': {
        'L_FORCE': 8.0,
        'P_REVERSAL': 0.08,      
        'THRESHOLD_PCT': 0.45,
        'CONV_THRESH': {
            'FORTE': 1.0,
            'MEDIA': 0.6,
            'FRACA': 0.20
        }
    },

    'DESCENDENTE_FRACA': {
        'L_FORCE': 2.0,
        'P_REVERSAL': 0.20,
        'THRESHOLD_PCT': 0.55,
        'CONV_THRESH': {
            'FORTE': 1.0,
            'MEDIA': 0.6,
            'FRACA': 0.10
        }
    },

    'DESCENDENTE_MEDIA': {
        'L_FORCE': 4.5,
        'P_REVERSAL': 0.15,
        'THRESHOLD_PCT': 0.50,
        'CONV_THRESH': {
            'FORTE': 1.0,
            'MEDIA': 0.6,
            'FRACA': 0.15
        }
    },

    'DESCENDENTE_FORTE': {
        'L_FORCE': 8.0,
        'P_REVERSAL': 0.08,
        'THRESHOLD_PCT': 0.45,
        'CONV_THRESH': {
            'FORTE': 1.0,
            'MEDIA': 0.6,
            'FRACA': 0.2
        }
    }
}

# PARÂMETROS INTERNOS
BAR_COLUMNS = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

# Mapeamento de Cores para as novas convicções
COLOR_MAP = {
    'ASCENDENTE_FORTE': 'blue',
    'DESCENDENTE_FORTE': 'red',
    'ASCENDENTE_MEDIA': 'slateblue',
    'DESCENDENTE_MEDIA': 'firebrick',
    'ASCENDENTE_FRACA': 'lightskyblue',
    'DESCENDENTE_FRACA': 'lightcoral',
    'LATERAL': 'green',
    'default': 'gray'
}


# LEITURA E PREPARAÇÃO DOS DADOS

def prepare_data(file_path: str) -> pd.DataFrame:
    """ Lê o arquivo de barras e prepara as colunas de cor e neutralidade. """
    print(f"1. Lendo o arquivo de barras: {file_path}")

    try:
        # Tenta ler como CSV (mais comum em ambiente Jupyter/Colab)
        if file_path.endswith('.csv') or 'WDO_ultimos_5dias.csv' in file_path:
             df = pd.read_csv(file_path, sep=',', header=0, names=BAR_COLUMNS, parse_dates=['Time'])
        # Tenta ler como XLSX
        elif file_path.endswith('.xlsx'):
             df = pd.read_excel(file_path, header=0, names=BAR_COLUMNS, parse_dates=['Time'])
        else:
             print("Erro: Tipo de arquivo não suportado (.csv ou .xlsx).")
             return pd.DataFrame()
    except Exception as e:
        print(f"ERRO ao ler o arquivo: {e}. Tentando formato alternativo ou verifique o caminho.")
        return pd.DataFrame()

    # Garantia de Float64 para precisão
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce', downcast='float')

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)

    # TRUE RANGE (para ATR)
    df['Prev_Close'] = df['Close'].shift(1)

    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            (df['High'] - df['Prev_Close']).abs(),
            (df['Low'] - df['Prev_Close']).abs()
        )
    )

    # Primeiro candle não tem close anterior
    df['TR'].fillna(df['High'] - df['Low'], inplace=True)

    # Cria novas colunas de cor (Excluindo neutro) e neutralidade
    df['Color_Up']   = (df['Close'] > df['Open']).astype(bool)
    df['Color_Down'] = (df['Close'] < df['Open']).astype(bool)
    df['Is_Neutral'] = (df['Close'] == df['Open']).astype(bool) # Novo: Candle neutro

    return df.copy()

def get_max_conviction(c1, c2):
    """ Retorna a convicção mais forte entre duas strings de convicção. """
    conviction_order = {'LATERAL': 0, 'FRACA': 1, 'MEDIA': 2, 'FORTE': 3}
    if conviction_order[c1] > conviction_order[c2]:
        return c1
    return c2


def is_structural_pullback(s1, s2, s3):
    """
    Detecta pullback estrutural:
    S1 e S3 na mesma direção, S2 contrário e curto,
    com dominância temporal clara.
    """

    # 1. Apenas segmentos direcionais
    if s1['sentido'] == 'LATERAL' or s3['sentido'] == 'LATERAL':
        return False

    # 2. Direção coerente
    if s1['sentido'] != s3['sentido']:
        return False

    # 3. S2 precisa ser oposto
    if s2['sentido'] == s1['sentido']:
        return False

    # 4. Pullback curto
    if s2['N'] > MAX_PULLBACK_N * 1.5:
        return False

    # 5. Dominância temporal
    if (s1['N'] + s3['N']) < 2 * s2['N']:
        return False

    # 6. Não deixar FRACO absorver FORTE
    order = {'LATERAL': 0, 'FRACA': 1, 'MEDIA': 2, 'FORTE': 3}
    if order[s2['conviction']] > order[s1['conviction']]:
        return False

    return True

def plot_todos_os_dias(
    ohlcv_df,
    df_segmentos_finais,
    hora_inicio="09:00",
    hora_fim="18:00"
):
    """
    Plota um gráfico por dia disponível no dataframe,
    empilhados verticalmente (um abaixo do outro).
    """

    dias = ohlcv_df['Time'].dt.date.unique()

    print(f"\nPlotando {len(dias)} dia(s): {dias}")

    for i, dia in enumerate(dias, start=1):

        print(f"\n--- Plotando dia {i}: {dia} ---")

        # Candles do dia no intervalo horário
        df_dia = ohlcv_df[
            (ohlcv_df['Time'].dt.date == dia) &
            (ohlcv_df['Time'].dt.time >= pd.to_datetime(hora_inicio).time()) &
            (ohlcv_df['Time'].dt.time <= pd.to_datetime(hora_fim).time())
        ].copy()

        if df_dia.empty:
            print("⚠️ Nenhum candle neste intervalo, pulando dia.")
            continue

        # Índices reais no dataframe original
        idx_inicio = df_dia.index.min()
        idx_fim    = df_dia.index.max()

        # Segmentos que cruzam este dia
        df_segmentos_dia = df_segmentos_finais[
            (df_segmentos_finais['idx2'] >= idx_inicio) &
            (df_segmentos_finais['idx1'] <= idx_fim)
        ].copy()

        if df_segmentos_dia.empty:
            print("⚠️ Nenhum segmento neste dia, pulando.")
            continue

        # Ajuste de índices para o dataframe diário
        df_segmentos_dia['idx1'] -= idx_inicio
        df_segmentos_dia['idx2'] -= idx_inicio

        # CLIP 
        n = len(df_dia)

        df_segmentos_dia['idx1'] = df_segmentos_dia['idx1'].clip(0, n - 1)
        df_segmentos_dia['idx2'] = df_segmentos_dia['idx2'].clip(0, n - 1)

        df_segmentos_dia = df_segmentos_dia[
            df_segmentos_dia['idx1'] <= df_segmentos_dia['idx2']
        ]

        # Plot
        plot_segmentation_candle_equiv(
            df_dia,
            df_segmentos_dia
        )

def reclassify_lateral_by_net_move(df, alpha=0.9, beta=0.55):
    """
    Reavalia segmentos LATERAIS consolidados de forma CONSERVADORA.
    Só reclassifica se houver deslocamento líquido dominante
    em relação à amplitude interna.
    """

    df = df.copy()

    for i, s in df.iterrows():

        if s['conviction'] != 'LATERAL':
            continue

        atr = s.get('ATR', None)
        amp = s.get('True_Amplitude', None)

        if atr is None or atr == 0 or amp is None or amp == 0:
            continue

        net_move = s['Close'] - s['Open']
        abs_move = abs(net_move)

        # Força mínima via PARAM_MATRIX (FRACA)
        params = PARAM_MATRIX['ASCENDENTE_FRACA']
        fraca_thresh = params['L_FORCE'] * params['CONV_THRESH']['FRACA']

        strength_atr = abs_move / atr
        if strength_atr < fraca_thresh * alpha:
            continue

        # Dominância sobre a amplitude (anti vai-e-volta)
        dominance = abs_move / amp
        if dominance < beta:
            continue

        # Respeito ao regime LATERAL
        lateral_params = PARAM_MATRIX['LATERAL']
        if dominance < lateral_params['THRESHOLD_PCT']:
            continue

        # reclassifica como FRACA
        sentido = 'ASCENDENTE' if net_move > 0 else 'DESCENDENTE'

        df.at[i, 'sentido'] = sentido
        df.at[i, 'conviction'] = 'FRACA'
        df.at[i, 'cor_segment'] = f"{sentido}_FRACA"

        print(
            f"--> LATERAL {s['idx1']}-{s['idx2']} "
            f"→ {sentido}_FRACA | "
            f"Δ={net_move:.2f} ATR={atr:.2f} AMP={amp:.2f}"
        )

    return df


def safe_to_absorb(target, small):
    """
    Verifica se a absorção é segura:
    - não inverte sentido
    - não reduz deslocamento líquido
    """

    # simula a junção
    merged = join_segments(target, small)

    # 1. sentido não pode mudar
    if merged['sentido'] != target['sentido']:
        return False

    # 2. deslocamento não pode diminuir demais
    old_move = abs(target['Close'] - target['Open'])
    new_move = abs(merged['Close'] - merged['Open'])

    if new_move < 0.7 * old_move:
        return False

    return True

def absorb_small_segments(df, min_n=4):
    """
    Remove segmentos pequenos SEM criar estados ilegais.
    """

    if df.empty or len(df) < 2:
        return df

    df = df.copy().reset_index(drop=True)
    i = 0

    while i < len(df):
        s = df.loc[i]

        if s['N'] >= min_n:
            i += 1
            continue

        sentido = s['sentido']

        prev_idx = i - 1 if i > 0 else None
        next_idx = i + 1 if i < len(df) - 1 else None

        # REGRA EXTRA: lateral curto entre mesma direção
        if s['sentido'] == 'LATERAL' and s['N'] < min_n:
            if prev_idx is not None and next_idx is not None:
                prev = df.loc[prev_idx]
                nxt  = df.loc[next_idx]

                if (
                    prev['sentido'] == nxt['sentido']
                    and prev['sentido'] != 'LATERAL'
                ):
                    # absorve no mais forte
                    target_idx = prev_idx if prev['N'] >= nxt['N'] else next_idx
                    target = df.loc[target_idx]

                    if safe_to_absorb(target, s):
                        df.loc[target_idx] = join_segments(target, s)
                        df = df.drop(i).reset_index(drop=True)
                        i -= 1
                        continue

        candidates = []

        # vizinho anterior
        if prev_idx is not None:
            prev = df.loc[prev_idx]
            if prev['sentido'] == sentido and prev['conviction'] != 'LATERAL':
                if safe_to_absorb(prev, s):
                    candidates.append(('prev', prev))

        # vizinho posterior
        if next_idx is not None:
            nxt = df.loc[next_idx]
            if nxt['sentido'] == sentido and nxt['conviction'] != 'LATERAL':
                if safe_to_absorb(nxt, s):
                    candidates.append(('next', nxt))

        
        # CASO 1 — existe absorção segura
        if candidates:
            side, target = max(candidates, key=lambda x: x[1]['N'])

            if side == 'prev':
                df.loc[prev_idx] = join_segments(target, s)
                df = df.drop(i).reset_index(drop=True)
                i -= 1
            else:
                df.loc[next_idx] = join_segments(target, s)
                df = df.drop(i).reset_index(drop=True)

            continue

        
        # CASO 2 — NÃO absorve → força lateral puro      
        df.loc[i, 'sentido'] = 'LATERAL'
        df.loc[i, 'conviction'] = 'LATERAL'
        df.loc[i, 'cor_segment'] = 'LATERAL'
        i += 1

    return df


def collapse_consecutive_laterals(df):
    """
    Garante que laterais consecutivos colapsem em um único segmento.
    Última etapa do pipeline.
    """
    if df.empty:
        return df

    collapsed = [df.iloc[0].copy()]

    for i in range(1, len(df)):
        prev = collapsed[-1]
        curr = df.iloc[i]

        if prev['conviction'] == 'LATERAL' and curr['conviction'] == 'LATERAL':
            collapsed[-1] = join_segments(prev, curr)
        else:
            collapsed.append(curr.copy())

    return pd.DataFrame(collapsed)

def apply_exhaustion_filter(df_segmentos):
    """
    Reclassifica segmentos como LATERAL quando há exaustão clara:
    FORTE → MÉDIA → FRACA na mesma direção.
    Atua SOMENTE após a consolidação completa.
    """

    df = df_segmentos.copy().reset_index(drop=True)

    conviction_rank = {
        'LATERAL': 0,
        'FRACA': 1,
        'MEDIA': 2,
        'FORTE': 3
    }

    for i in range(2, len(df)):
        s0 = df.loc[i-2]
        s1 = df.loc[i-1]
        s2 = df.loc[i]

        # Mesma direção e direcional
        if (
            s0['sentido'] == s1['sentido'] == s2['sentido']
            and s2['sentido'] != 'LATERAL'
        ):
            r0 = conviction_rank[s0['conviction']]
            r1 = conviction_rank[s1['conviction']]
            r2 = conviction_rank[s2['conviction']]

            # Exaustão clara
            if r0 > r1 > r2:
                df.at[i, 'sentido'] = 'LATERAL'
                df.at[i, 'conviction'] = 'LATERAL'
                df.at[i, 'cor_segment'] = 'LATERAL'

    return df

def absorption_multiplier(conviction):
    return {
        'FRACA': 1.0,
        'MEDIA': 0.7,
        'FORTE': 0.4
    }.get(conviction, 1.0)

def get_stronger_segment(s1, s2):
    order = {'LATERAL': 0, 'FRACA': 1, 'MEDIA': 2, 'FORTE': 3}
    return s1 if order[s1['conviction']] >= order[s2['conviction']] else s2

def classify_conviction(price_change, sentido, segment_stub=None):
    """
    Classifica convicção usando exclusivamente o PARAM_MATRIX.
    """
    # Garantia de stub
    if segment_stub is None:
        segment_stub = {'sentido': sentido, 'conviction': 'FRACA'}
    abs_change = abs(price_change)

    # Normalização por ATR 
    atr = segment_stub.get('ATR', None)
    if atr and atr > 0:
      abs_change = abs_change / atr

    # Segmento lateral explícito
    if sentido == 'LATERAL':
        return 'LATERAL'

    # Cria stub mínimo se não existir
    if segment_stub is None:
        segment_stub = {'sentido': sentido, 'conviction': 'FRACA'}

    params = get_params(segment_stub)
    L = params['L_FORCE']
    conv_thresh = params['CONV_THRESH']

    if abs_change >= L * conv_thresh['FORTE']:
      return 'FORTE'
    elif abs_change >= L * conv_thresh['MEDIA']:
      return 'MEDIA'
    elif abs_change >= L * conv_thresh['FRACA']:
      return 'FRACA'
    else:
      return 'LATERAL'

def get_params(segment):
    """
    Retorna os parâmetros corretos para o segmento,
    normalizando estados ilegais da máquina de estados.
    """

    sentido = segment.get('sentido')
    conviction = segment.get('conviction')

    # Qualquer coisa que envolva LATERAL usa o regime LATERAL puro
    if sentido == 'LATERAL' or conviction == 'LATERAL':
        return PARAM_MATRIX['LATERAL']

    key = f"{sentido}_{conviction}"

    if key not in PARAM_MATRIX:
        return PARAM_MATRIX['LATERAL']

    key = f"{sentido}_{conviction}"
    return PARAM_MATRIX[key]

def has_meaningful_displacement(s):
    """
    Exige deslocamento mínimo coerente com a convicção do segmento.
    """

    net_move = abs(s['Close'] - s['Open'])
    atr = s.get('ATR', None)

    if atr is None or atr == 0:
        return False

    thresholds = {
        'FRACA': 0.5,
        'MEDIA': 0.8,
        'FORTE': 1.2
    }

    threshold = thresholds.get(s['conviction'], 0.8)

    return (net_move / atr) >= threshold

def imprimeSlice(pure_slice, N_pure, open_i_pure, close_f_pure, price_change_pure, ratio_pure, true_amplitude, conviction, sentido, cor_segment):
    print("***************")
    print("Slice Atual:")
    print(pure_slice)
    print(N_pure)
    print(open_i_pure)
    print(close_f_pure)
    print(price_change_pure)
    print(ratio_pure)
    print(true_amplitude)
    print(conviction)
    print(sentido)
    print(cor_segment)
    print("***************")

def join_segments(s1, s2):
    """
    Junta s2 em s1 SEM alterar os originais.
    Retorna um novo segmento.
    """
    s = s1.copy()

    s['idx2'] = s2['idx2']
    s['Close'] = s2['Close']
    s['N'] = s1['N'] + s2['N']

    price_change = s['Close'] - s['Open']
    s['Ratio'] = price_change / s['N']

    # True Amplitude
    s['True_Amplitude'] = max(
        s1.get('True_Amplitude', 0),
        s2.get('True_Amplitude', 0)
    )

    # Convicção mais forte
    s['conviction'] = get_max_conviction(
        s1['conviction'],
        s2['conviction']
    )

    # ATR ponderado
    if 'ATR' in s1 and 'ATR' in s2:
        s['ATR'] = (
            s1['ATR'] * s1['N'] +
            s2['ATR'] * s2['N']
        ) / s['N']

    # REAVALIA O SENTIDO APÓS A JUNÇÃO
    net_change = s['Close'] - s['Open']

    if net_change > 0:
        s['sentido'] = 'ASCENDENTE'
    elif net_change < 0:
        s['sentido'] = 'DESCENDENTE'
    else:
        s['sentido'] = 'LATERAL'

    # Ajusta cor do segmento
    if s['sentido'] == 'LATERAL' or s['conviction'] == 'LATERAL':
        s['cor_segment'] = 'LATERAL'
    else:
        s['cor_segment'] = f"{s['sentido']}_{s['conviction']}"

    return s

# CRIA BLOCOS EQUIVALENTES
def create_equivalent_candles(df):
    """
    Cria CandleEquivalente PURO (no primeiro estágio) com convicção 100% paramétrica.
    Permite junção de candles neutros na sequência pura.
    """
    equivalent_list = []

    i = 0
    N = len(df)

    while i < N:
        # 1 Pular candles neutros iniciais
        j = i
        while j < N and df.iloc[j]['Is_Neutral']:
            j += 1

        if j == N:
            break

        start_index = j
        i = j

        is_up_pure = df.iloc[i]['Color_Up']

        # 2️ Estender sequência pura (permitindo neutros)
        j = i + 1
        while j < N:
            if df.iloc[j]['Is_Neutral']:
                j += 1
                continue

            if (is_up_pure and df.iloc[j]['Color_Up']) or \
               (not is_up_pure and df.iloc[j]['Color_Down']):
                j += 1
            else:
                break

        end_index_pure = j - 1

        # 3 Slice puro
        pure_slice = df.iloc[start_index:end_index_pure + 1]

        # 4 Métricas básicas
        N_pure = len(pure_slice)
        # Peso temporal médio (recência)
        # candles mais recentes têm mais peso
        weights = np.arange(1, N_pure + 1)
        temporal_weight = weights.mean() / N_pure
        open_i_pure = pure_slice['Open'].iloc[0]
        close_f_pure = pure_slice['Close'].iloc[-1]
        price_change_pure = close_f_pure - open_i_pure

        # 5 Amplitude
        true_amplitude = pure_slice['High'].max() - pure_slice['Low'].min()
        atr_local = pure_slice['TR'].mean()

        price_change_pure = close_f_pure - open_i_pure

        # 6 Sentido (ORIGINAL — MANTER)
        if price_change_pure > 0:
          sentido = 'ASCENDENTE'
        elif price_change_pure < 0:
          sentido = 'DESCENDENTE'
        else:
          sentido = 'LATERAL'

        # 7 Convicção 100% paramétrica
        segment_stub = {
          'sentido': sentido,
          'conviction': 'FRACA' if sentido != 'LATERAL' else 'LATERAL',
          'ATR': atr_local
        }

        conviction = classify_conviction(
          price_change_pure,
          sentido,
          segment_stub
        )

        equiv_type = 'PURO'
        cor_segment = (
            f"{sentido}_{conviction}"
            if conviction != 'LATERAL'
            else 'LATERAL'
        )

        # 8 Append final
        equivalent_list.append({
            'type': equiv_type,
            'sentido': sentido,
            'conviction': conviction,
            'cor_segment': cor_segment,
            'idx1': start_index,
            'idx2': end_index_pure,
            'Open': open_i_pure,
            'Close': close_f_pure,
            'N': N_pure,
            'Ratio': price_change_pure / N_pure,
            'True_Amplitude': true_amplitude,
            'ATR': atr_local,
            'W_TIME': temporal_weight,
        })

        i = end_index_pure + 1

    return equivalent_list



# JUNÇÃO DE CANDLE EQUIVALENTE (JUNÇÃO DESABILITADA)

def join_equivalent_candles(equiv_list):
    """
    Função de junção com Regras 1 (Lateral), 3 (Mesma Direção),
    RAL (Absorção Lateral Curta) e AF (Absorção de Fraco por Lateral).
    Possui 'prints' de debug para rastrear a junção.
    """
    prev_segment = None
    def conviction_rank(conv):
        return {
         'LATERAL': 0,
         'FRACA': 1,
         'MEDIA': 2,
         'FORTE': 3
        }.get(conv, 0)

    final_segments = []
    if not equiv_list:
        return pd.DataFrame()

    current_segment = equiv_list[0].copy()
    buffer_segment = None
    AMPLITUDE_MAX_DIFF = 1.5


    for i in range(1, len(equiv_list)):
        next_equiv = equiv_list[i]

        # ABSORÇÃO DE PULLBACK ESTRUTURAL (S1–S2–S3)
        if i < len(equiv_list) - 1:
          s1 = current_segment
          s2 = next_equiv
          s3 = equiv_list[i + 1]

          if is_structural_pullback(s1, s2, s3):
            # Absorve o pullback (S2) dentro de S1
            current_segment = join_segments(current_segment, s2)

            # Trava de deslocamento líquido
            if not has_meaningful_displacement(current_segment):
                 current_segment['sentido'] = 'LATERAL'
                 current_segment['conviction'] = 'LATERAL'
                 current_segment['cor_segment'] = 'LATERAL'

            continue

        if buffer_segment:
             print(f"BUFFER PENDENTE (S2): {buffer_segment['conviction']} N={buffer_segment['N']}")

        # A. PROCESSAMENTO DE BUFFER PENDENTE (RAL)
        if buffer_segment:
            s1_sentido = current_segment['sentido'] # Sentido do segmento antes do buffer (S1)

            # 1A. SUCESSO RAL: S3 é direcional e tem o mesmo sentido de S1
            is_coherent_s3 = (next_equiv['conviction'] != 'LATERAL' and
                              next_equiv['sentido'] == s1_sentido)

            if is_coherent_s3:
                # Junção tripla (S1 + S2 + S3)
                # O S1 (current) já está pronto. Junta S2 e depois S3
                current_segment = join_segments(current_segment, buffer_segment)
                current_segment = join_segments(current_segment, next_equiv)

                print(f"--> RAL SUCESSO: Junção tripla concluída. Novo N={current_segment['N']}")

                buffer_segment = None # Limpa o buffer: sucesso na junção tripla
                continue # Vai para o próximo next_equiv, pois este já foi processado

            # 1B. FALHA RAL: S3 não é coerente ou é Lateral
            else:
                # Finaliza S1, S2 (buffer) vira o novo current_segment
                final_segments.append(current_segment)
                current_segment = buffer_segment.copy()
                buffer_segment = None

                print("--> RAL FALHA: S1 finalizado. S2 (buffer) é o novo current.")
                # O next_equiv (S3) será avaliado contra o S2 (novo current) nas regras abaixo (C).

        # B. CANDIDATO A BUFFER (S2)
        params_lateral = get_params({'sentido': 'LATERAL', 'conviction': 'LATERAL'})

        is_ral_candidate = (
          current_segment['conviction'] in ['MEDIA', 'FORTE'] and
          next_equiv['conviction'] == 'LATERAL' and
          next_equiv['N'] <= params_lateral['RAL_MAX_N']
        )

        if not buffer_segment and is_ral_candidate:
            buffer_segment = next_equiv.copy()
            print("--> BUFFER ATIVADO: LATERAL CURTO ARMAZENADO (RAL)")
            continue


        # C. REGRAS DE JUNÇÃO REGULARES 
        can_join = False

        # 1. REGRA 1: LATERAL + LATERAL
        if (current_segment['conviction'] == 'LATERAL' and
          next_equiv['conviction'] == 'LATERAL'):

          params = get_params(current_segment)

          lateral_range = current_segment['True_Amplitude']
          next_move = abs(next_equiv['Close'] - current_segment['Close'])

          if lateral_range == 0:
           can_join = True
          else:
           ratio = next_move / lateral_range
           can_join = ratio <= params['THRESHOLD_PCT']

        # 2. REGRA 2: MESMA DIREÇÃO (Consolidação de tendência)
        elif (current_segment['sentido'] == next_equiv['sentido'] and
                 current_segment['conviction'] != 'LATERAL' and
                 next_equiv['conviction'] != 'LATERAL'):

             stronger = get_stronger_segment(current_segment, next_equiv)
             params = get_params(stronger)

             atr = current_segment.get('ATR', None)

             base_move = abs(current_segment['Close'] - current_segment['Open'])
             reversal_move = abs(next_equiv['Open'] - current_segment['Close'])

             if atr and atr > 0:
              base_move /= atr
              reversal_move /= atr

             # Proteção contra divisão por zero
             if base_move == 0:
              can_join = True
             else:
              reversal_ratio = reversal_move / base_move
              w = current_segment.get('W_TIME', 1.0)
              can_join = reversal_ratio <= params['P_REVERSAL'] * w
            # print("--> REGRA 3 (MESMA DIREÇÃO) ATIVA")

        # 3. REGRA 3 (AF): ABSORÇÃO DE FRACO POR LATERAL

        # 3a. Segmento Lateral (Dominante, N > 2) + Segmento Direcional Curto (Candidato, N <= 2)
        elif (current_segment['conviction'] == 'LATERAL' and current_segment['N'] > 2 and
          next_equiv['conviction'] != 'LATERAL' and next_equiv['N'] <= 2):

          params = get_params(current_segment)

          next_strength = abs(next_equiv['Close'] - next_equiv['Open'])
          atr = current_segment.get('ATR', None)

          if atr and atr > 0:
           next_strength /= atr

          mult = absorption_multiplier(next_equiv['conviction'])
          w = current_segment.get('W_TIME', 1.0)
          can_join = next_strength <= params['L_FORCE'] * mult * w
                # print(f"--> REGRA 4.3a (LATERAL + FRACO) ATIVA. Diff Amp: {abs(abs_change_C - abs_change_L):.2f}")

        # 3b. Segmento Direcional Curto (Candidato, N <= 2) + Segmento Lateral (Dominante, N > 2)
        elif (current_segment['conviction'] != 'LATERAL' and current_segment['N'] <= 2 and
           next_equiv['conviction'] == 'LATERAL' and next_equiv['N'] > 2):

           params = get_params(current_segment)

           reversal_strength = abs(next_equiv['Close'] - current_segment['Close'])

           if reversal_strength <= params['L_FORCE']:

             next_equiv = join_segments(current_segment, next_equiv)

             current_segment['conviction'] = 'LATERAL'
             current_segment['cor_segment'] = 'LATERAL'

             current_segment = next_equiv.copy()

             continue

        # 4. DEFAULT: Não junta
        else:
            can_join = False


        if can_join:
            # AÇÃO DE JUNÇÃO REGULAR 

            is_af_lateral_plus_weak = (current_segment['conviction'] == 'LATERAL' and next_equiv['conviction'] != 'LATERAL')

            current_segment = join_segments(current_segment, next_equiv)

            if is_af_lateral_plus_weak:
                 # Força a convicção para LATERAL na junção 3a (Lateral + Fraco)
                 current_segment['conviction'] = 'LATERAL'
                 current_segment['cor_segment'] = 'LATERAL'


        else:
            # AÇÃO DE FINALIZAÇÃO: Finaliza o segmento atual e começa o próximo
            final_segments.append(current_segment)
            current_segment = next_equiv.copy()
        prev_segment = final_segments[-1].copy() if final_segments else None

    # D. FINALIZAÇÃO DO ÚLTIMO SEGMENTO E BUFFER ---

    # Adiciona o último segmento que ficou no current_segment ou no buffer
    if buffer_segment:
        print("\nFINALIZAÇÃO: Adicionando segmento e buffer pendente.")
        final_segments.append(current_segment)
        final_segments.append(buffer_segment)
    else:
        print("\nFINALIZAÇÃO: Adicionando o último segmento.")
        final_segments.append(current_segment)

    return pd.DataFrame(final_segments)

def apply_directional_consistency_filter(df_ohlc_bruto, equivalent_list):
    """
    Reclassifica segmentos LATERALS se a maioria dos candles (>= threshold_pct)
    apresentar consistência direcional no Low (para subida) ou High (para descida).
    """


    for segment in equivalent_list:
        if segment['conviction'] == 'LATERAL':
            params = get_params(segment)
            L_FRACA = params['L_FORCE'] * 0.3
            threshold = params['THRESHOLD_PCT']

            # 1. Extrair os dados brutos (OHLC) do segmento
            start_idx = segment['idx1']
            end_idx = segment['idx2']
            segment_data = df_ohlc_bruto.iloc[start_idx : end_idx + 1]

            # Se N=1 ou N=0, não é possível avaliar a consistência sequencial
            if len(segment_data) <= 1:
                continue

            # 2. Calcular a consistência

            n = len(segment_data)
            weights = np.linspace(0.3, 1.0, n)  # mais peso no final

            # Consistência de subida (Low crescente)
            up_flags = (segment_data['Low'] > segment_data['Low'].shift(1)).fillna(False)
            up_score = np.sum(weights * up_flags)

            # Consistência de descida (High decrescente)
            down_flags = (segment_data['High'] < segment_data['High'].shift(1)).fillna(False)
            down_score = np.sum(weights * down_flags)

            max_score = weights.sum()
            up_ratio = up_score / max_score
            down_ratio = down_score / max_score

            # 3. Aplicar filtro de consistência direcional (paramétrico)
            if up_ratio >= threshold and segment['N'] > 1:

               segment['sentido'] = 'ASCENDENTE'

               new_conviction = classify_conviction(
                segment['Close'] - segment['Open'],
                segment['sentido'],
                segment
               )

               segment['conviction'] = new_conviction
               segment['cor_segment'] = f"{segment['sentido']}_{segment['conviction']}"

               print(f"--> FILTRO ATIVADO: Segmento Lateral {start_idx}-{end_idx} reclassificado para {segment['cor_segment']} (Low Consistência: {up_ratio:.2f})")

            elif down_ratio >= threshold and segment['N'] > 1:

               segment['sentido'] = 'DESCENDENTE'

               new_conviction = classify_conviction(
                segment['Close'] - segment['Open'],
                segment['sentido'],
                segment
               )

               segment['conviction'] = new_conviction
               segment['cor_segment'] = f"{segment['sentido']}_{segment['conviction']}"

               print(f"--> FILTRO ATIVADO: Segmento Lateral {start_idx}-{end_idx} reclassificado para {segment['cor_segment']} (High Consistência: {down_ratio:.2f})")

    return equivalent_list

# FUNÇÃO 4: PLOTAGEM MATPLOTLIB PURO 
def plot_segmentation_candle_equiv(df_ohlc, df_segmentos):
    """
    Plota OHLC + segmentação para UM ÚNICO DIA,
    usando horário real no eixo X.
    """

    print("Gerando gráfico do primeiro dia (eixo horário)...")

    df = df_ohlc.copy()

    # Garantia de datetime
    df['Time'] = pd.to_datetime(df['Time'])

    df['Direction'] = np.where(df['Close'] > df['Open'], 'Up', 'Down')
    df['Color'] = np.where(df['Direction'] == 'Up', 'blue', 'red')
    df['Body_Height'] = (df['Close'] - df['Open']).abs()
    df['Body_Bottom'] = np.where(
        df['Direction'] == 'Up',
        df['Open'],
        df['Close']
    )

    df = df.set_index('Time')
    dates_numeric = mdates.date2num(df.index)

    fig, ax = plt.subplots(figsize=(18, 10))

    # Segmentos
    for _, row in df_segmentos.iterrows():
        idx1 = row['idx1']
        idx2 = row['idx2']

        t1 = df.index[idx1]
        t2 = df.index[idx2]

        ax.axvspan(
            t1,
            t2 + (df.index[1] - df.index[0]),
            color=COLOR_MAP.get(row['cor_segment'], COLOR_MAP['default']),
            alpha=0.15,
            zorder=1
        )

        ax.plot(
            [t1, t2],
            [row['Open'], row['Close']],
            color=COLOR_MAP.get(row['cor_segment'], COLOR_MAP['default']),
            linewidth=3,
            linestyle='--',
            zorder=4,
            alpha=0.7
        )

    # Candles
    ax.vlines(
        dates_numeric,
        df['Low'],
        df['High'],
        color='black',
        linewidth=1,
        zorder=2
    )

    time_diff = (df.index[1] - df.index[0]).total_seconds()
    width = (time_diff * 0.8) / (24 * 60 * 60)

    ax.bar(
        dates_numeric,
        df['Body_Height'],
        width=width,
        bottom=df['Body_Bottom'],
        color=df['Color'],
        edgecolor='black',
        linewidth=0.5,
        zorder=3
    )

    # Legenda
    legend_patches = [
        Rectangle((0, 0), 1, 1, fc=COLOR_MAP['ASCENDENTE_FORTE'], alpha=0.3, label='ASC (FORTE)'),
        Rectangle((0, 0), 1, 1, fc=COLOR_MAP['DESCENDENTE_FORTE'], alpha=0.3, label='DESC (FORTE)'),
        Rectangle((0, 0), 1, 1, fc=COLOR_MAP['ASCENDENTE_MEDIA'], alpha=0.3, label='ASC (MÉDIA)'),
        Rectangle((0, 0), 1, 1, fc=COLOR_MAP['DESCENDENTE_MEDIA'], alpha=0.3, label='DESC (MÉDIA)'),
        Rectangle((0, 0), 1, 1, fc=COLOR_MAP['ASCENDENTE_FRACA'], alpha=0.3, label='ASC (FRACA)'),
        Rectangle((0, 0), 1, 1, fc=COLOR_MAP['DESCENDENTE_FRACA'], alpha=0.3, label='DESC (FRACA)'),
        Rectangle((0, 0), 1, 1, fc=COLOR_MAP['LATERAL'], alpha=0.3, label='LATERAL'),
    ]

    ax.legend(handles=legend_patches, loc='lower right')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title('Segmentação por Candle Equivalente – Primeiro Dia')
    ax.set_ylabel('Preço')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()





# MAIN
if __name__ == '__main__':
    print("--- INICIANDO PROCESSAMENTO (Candle Equivalente) ---")

    # Leitura e preparação dos dados
    ohlcv_df = prepare_data(BAR_FILE_PATH)

    if ohlcv_df.empty:
        print("\nProcessamento interrompido.")
    else:
        # 1 Criação dos Candle Equivalentes (dados completos)
        equiv_list = create_equivalent_candles(
            ohlcv_df
        )
        print("\nSANITY 1 — CONVICÇÕES INICIAIS")
        for s in equiv_list[:10]:
          print(
           f"{s['sentido']:11} | "
           f"N={s['N']:2} | "
           f"Δ={s['Close'] - s['Open']:.2f} | "
           f"{s['conviction']}"
           )

        print(f"1. Criados {len(equiv_list)} CandleEquivalentes iniciais.")

        # 2 Filtro de consistência direcional
        equiv_list = apply_directional_consistency_filter(
            ohlcv_df,
            equiv_list
        )

        # 3 Junção / consolidação
        df_segmentos_finais = join_equivalent_candles(equiv_list)
        # tamanho mínimo de segmento
        df_segmentos_finais = absorb_small_segments(
        df_segmentos_finais,
        min_n=4
        )

        df_segmentos_finais = apply_exhaustion_filter(df_segmentos_finais)
        df_segmentos_finais = collapse_consecutive_laterals(df_segmentos_finais)
        df_segmentos_finais = reclassify_lateral_by_net_move(
          df_segmentos_finais,
          alpha=0.9,
          beta=0.55
        )

        print(f"2. Consolidados em {len(df_segmentos_finais)} segmentos finais.")

        plot_todos_os_dias(
          ohlcv_df,
          df_segmentos_finais
        )

    print("--- PROCESSAMENTO CONCLUÍDO ---")

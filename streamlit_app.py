# streamlit_app.py — Análise de Flexão em Vigas (Biaxial) | Streamlit Single-File Multipage
# UI profissional + páginas internas (sidebar) + gráficos Plotly
# Modelo: Euler–Bernoulli + MEF 1D (flexão em 2 planos independentes)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# CONFIG (APENAS UMA VEZ, NO TOPO)
# ============================================================
st.set_page_config(
    page_title="Análise de Flexão em Vigas (Biaxial)",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# AUTH (opcional)
# ============================================================
def require_auth():
    """
    Usa st.secrets["APP_PASSWORD"].
    - Streamlit Cloud: App -> Settings -> Secrets
    - Local: .streamlit/secrets.toml com:
      APP_PASSWORD="sua_senha"
    """
    if "APP_PASSWORD" not in st.secrets:
        return

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Acesso restrito")
        st.caption("Digite a senha para acessar a calculadora.")
        pwd = st.text_input("Senha", type="password")

        if st.button("Entrar", type="primary"):
            if pwd == st.secrets.get("APP_PASSWORD", ""):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Senha incorreta.")
        st.stop()

require_auth()

# ============================================================
# UNIDADES
# ============================================================
def mm_to_m(x_mm: float) -> float:
    return float(x_mm) / 1000.0

def m_to_mm(x_m: float) -> float:
    return float(x_m) * 1000.0

def MPa_to_Pa(x_MPa: float) -> float:
    return float(x_MPa) * 1e6
# ============================================================
# DADOS DEFAULT (materiais e catálogo mínimo)
# ============================================================

DEFAULT_MATS = {
    "aco_1020":   {"E": 210e9, "fy": 350e6, "nu": 0.30},
    "aco_1045":   {"E": 210e9, "fy": 530e6, "nu": 0.29},
    "astm_a36":   {"E": 200e9, "fy": 250e6, "nu": 0.26},
    "inox_304":   {"E": 193e9, "fy": 215e6, "nu": 0.29},
    "inox_316":   {"E": 193e9, "fy": 205e6, "nu": 0.29},
    "al_6061_t6": {"E":  69e9, "fy": 275e6, "nu": 0.33},
    "al_5052":    {"E":  70e9, "fy": 193e6, "nu": 0.33},
    "nylon_pa6":  {"E": 2.7e9, "fy":  70e6, "nu": 0.39},
    "pla":        {"E": 3.5e9, "fy":  60e6, "nu": 0.36},
}

DEFAULT_PROFILES = pd.DataFrame([
    {"family":"RETANGULO",        "name":"Ret 100 x 10",     "b_mm":100.0, "h_mm":10.0},
    {"family":"BARRA REDONDA",    "name":"Barra Ø20",        "d_mm":20.0},
    {"family":"TUBO REDONDO",     "name":"Tubo Ø60,3 x 3,0", "od_mm":60.3, "t_mm":3.0},
    {"family":"TUBO QUADRADO",    "name":"Tubo 50x50x3,0",   "b_mm":50.0,  "h_mm":50.0, "t_mm":3.0},
    {"family":"TUBO RETANGULAR",  "name":"Tubo 80x40x3,0",   "b_mm":80.0,  "h_mm":40.0, "t_mm":3.0},
])

# ============================================================
# STATE INIT
# ============================================================

def init_state():
    st.session_state.setdefault("materials", DEFAULT_MATS.copy())
    st.session_state.setdefault("profiles_df", DEFAULT_PROFILES.copy())
    st.session_state.setdefault("loads", [])
    st.session_state.setdefault("results", None)

    # modelo estrutural
    st.session_state.setdefault("unit_system", "mm (mm, N, MPa)")
    st.session_state.setdefault("L_in", 2000.0)
    st.session_state.setdefault("apoio_esq", "Apoio simples (v=0)")
    st.session_state.setdefault("apoio_dir", "Apoio simples (v=0)")
    st.session_state.setdefault("material", "aco_1020")
    st.session_state.setdefault("FS", 1.5)
    st.session_state.setdefault("lim_flecha", "L/250")
    st.session_state.setdefault("ne", 160)

    # seção
    st.session_state.setdefault("sec_desc", "RETANGULO | Ret 100 x 10")
    st.session_state.setdefault("Iy", 0.0)
    st.session_state.setdefault("Iz", 0.0)
    st.session_state.setdefault("yext", (0.0, 0.0))
    st.session_state.setdefault("zext", (0.0, 0.0))

init_state()
# ============================================================
# SEÇÕES: Iy/Iz (m^4) e extremos
# Convenção:
# - x ao longo da viga
# - seção: y (horizontal), z (vertical)
# ============================================================

def rect_Iy_Iz(b_m: float, h_m: float):
    Iy = (b_m * h_m**3) / 12.0
    Iz = (h_m * b_m**3) / 12.0
    yext = (-b_m/2, b_m/2)
    zext = (-h_m/2, h_m/2)
    return Iy, Iz, yext, zext

def round_solid_I(d_m: float):
    I = (np.pi/64.0) * d_m**4
    r = d_m/2
    yext = (-r, r)
    zext = (-r, r)
    return I, I, yext, zext

def round_tube_I(od_m: float, t_m: float):
    ro = od_m/2
    ri = max(ro - t_m, 0.0)
    di = 2*ri
    I = (np.pi/64.0) * (od_m**4 - di**4)
    yext = (-ro, ro)
    zext = (-ro, ro)
    return I, I, yext, zext

def rect_tube_I(b_m: float, h_m: float, t_m: float):
    bi = max(b_m - 2*t_m, 0.0)
    hi = max(h_m - 2*t_m, 0.0)
    Iy_out, Iz_out, _, _ = rect_Iy_Iz(b_m, h_m)
    if bi > 0 and hi > 0:
        Iy_in, Iz_in, _, _ = rect_Iy_Iz(bi, hi)
    else:
        Iy_in, Iz_in = 0.0, 0.0
    Iy = Iy_out - Iy_in
    Iz = Iz_out - Iz_in
    yext = (-b_m/2, b_m/2)
    zext = (-h_m/2, h_m/2)
    return Iy, Iz, yext, zext

# ============================================================
# Inicializa seção default (se ainda não calculada)
# ============================================================
if st.session_state.Iy <= 0 or st.session_state.Iz <= 0:
    Iy0, Iz0, yext0, zext0 = rect_Iy_Iz(mm_to_m(100.0), mm_to_m(10.0))
    st.session_state.Iy = Iy0
    st.session_state.Iz = Iz0
    st.session_state.yext = yext0
    st.session_state.zext = zext0
   # ============================================================
# FEM Euler–Bernoulli 1D
# DOF por nó: [v, theta]
# ============================================================

def beam_element_k(EI, Le):
    L = Le
    return (EI / L**3) * np.array([
        [ 12,   6*L, -12,   6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L,  12,  -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2],
    ], dtype=float)

def shape_N(Le, xi):
    L = Le
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L*(xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L*(-xi**2 + xi**3)
    return np.array([N1, N2, N3, N4], float)

def udl_equiv_nodal_load(w, Le):
    L = Le
    return np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12], dtype=float)

def apply_bc(K, F, fixed_dofs):
    all_dofs = np.arange(K.shape[0])
    free = np.setdiff1d(all_dofs, np.array(fixed_dofs, dtype=int))
    return free, K[np.ix_(free, free)], F[free]

def solve_beam_FEM(L, EI, apoio_esq, apoio_dir, loads, ne=160):
    """
    loads: lista de dicts:
      - {"type":"P", "x":..., "P":...}
      - {"type":"M", "x":..., "M":...}
      - {"type":"w", "a":..., "b":..., "w":...}
    retorna: xs, v(x), V(x), M(x), reactions(dof->val)
    """
    x_nodes = np.linspace(0.0, L, ne+1)
    Le = L/ne
    nn = ne+1
    ndof = 2*nn

    K = np.zeros((ndof, ndof), float)
    F = np.zeros(ndof, float)

    # rigidez global
    for e in range(ne):
        ke = beam_element_k(EI, Le)
        dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
        K[np.ix_(dofs, dofs)] += ke

    # UDL por elemento (aprox constante no elemento)
    for e in range(ne):
        x0 = x_nodes[e]
        x1 = x_nodes[e+1]
        w_elem = 0.0
        for ld in loads:
            if ld["type"] == "w":
                a = ld["a"]; b = ld["b"]; w = ld["w"]
                if (x1 > a) and (x0 < b):
                    w_elem += w
        if abs(w_elem) > 0:
            feq = udl_equiv_nodal_load(w_elem, Le)
            dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
            F[dofs] += feq

    # força concentrada consistente (interpolação por shape function)
    for ld in loads:
        if ld["type"] == "P":
            xP = float(np.clip(ld["x"], 0.0, L))
            P = ld["P"]
            e = min(int(xP / Le), ne-1)
            x0 = x_nodes[e]
            xi = (xP - x0) / Le
            N = shape_N(Le, xi)
            dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
            F[dofs] += P * N

        elif ld["type"] == "M":
            xM = float(np.clip(ld["x"], 0.0, L))
            M = ld["M"]
            i = int(round(xM / Le))
            i = int(np.clip(i, 0, nn-1))
            F[2*i + 1] += M

    # condições de contorno
    fixed = []
    if apoio_esq == "Engastado":
        fixed += [0, 1]
    elif apoio_esq == "Apoio simples (v=0)":
        fixed += [0]
    # Livre: nada

    last_v = 2*(nn-1)
    last_t = last_v + 1
    if apoio_dir == "Engastado":
        fixed += [last_v, last_t]
    elif apoio_dir == "Apoio simples (v=0)":
        fixed += [last_v]

    free, Kff, Ff = apply_bc(K, F, fixed)
    d = np.zeros(ndof, float)
    if len(free) > 0:
        d[free] = np.linalg.solve(Kff, Ff)

    Rfull = K @ d - F
    reactions = {int(fd): float(Rfull[fd]) for fd in fixed}

    # pós: montar V e M por elemento
    xs, vs, Ms, Vs = [], [], [], []
    for e in range(ne):
        dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
        de = d[dofs]
        ke = beam_element_k(EI, Le)

        w_elem = 0.0
        x0 = x_nodes[e]
        x1 = x_nodes[e+1]
        for ld in loads:
            if ld["type"] == "w":
                a = ld["a"]; b = ld["b"]; w = ld["w"]
                if (x1 > a) and (x0 < b):
                    w_elem += w
        feq = udl_equiv_nodal_load(w_elem, Le) if abs(w_elem) > 0 else np.zeros(4)

        fint = ke @ de - feq  # [V1, M1, V2, M2]
        V1, M1, V2, M2 = fint

        for xi in np.linspace(0, 1, 6, endpoint=False):
            xg = x_nodes[e] + xi*Le
            N = shape_N(Le, xi)
            vg = float(N @ de)
            xs.append(xg)
            vs.append(vg)
            Ms.append(M1*(1-xi) + M2*xi)
            Vs.append(V1*(1-xi) + V2*xi)

    xs.append(L)
    vs.append(d[last_v])
    Ms.append(Ms[-1] if Ms else 0.0)
    Vs.append(Vs[-1] if Vs else 0.0)

    return np.array(xs), np.array(vs), np.array(Vs), np.array(Ms), reactions

# ============================================================
# HELPERS: reações e preview (Plotly)
# ============================================================

def reactions_table(reac: dict) -> pd.DataFrame:
    rows = []
    for dof, val in sorted(reac.items()):
        node = dof // 2
        kind = "v (N)" if dof % 2 == 0 else "θ (N·m)"
        rows.append({"DOF": int(dof), "Nó": int(node), "Tipo": kind, "Reação": float(val)})
    return pd.DataFrame(rows)

def plot_preview_model(L_m: float, loads: list):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, L_m], y=[0, 0], mode="lines", name="Viga"))

    if loads:
        xs = [ld["x_m"] for ld in loads if "x_m" in ld]
        ys = [0 for _ in xs]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Carregamentos"))

    fig.update_layout(
        xaxis_title="x (m)",
        yaxis_visible=False,
        height=260,
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig 
# ============================================================
# HEADER
# ============================================================

c1, c2 = st.columns([0.12, 0.88], vertical_alignment="center")
with c1:
    try:
        st.image("assets/logo.png", width=72)
    except Exception:
        st.write("📐")
with c2:
    st.title("Análise de Flexão em Vigas (Biaxial)")
    st.caption("Modelo Euler–Bernoulli • MEF 1D • Saída principal em mm • Plotly")

st.divider()

# ============================================================
# NAVEGAÇÃO (Multipage dentro de 1 arquivo)
# ============================================================

with st.sidebar:
    st.header("Navegação")
    page = st.radio(
        "Página",
        [
            "Calculadora",
            "Propriedades da Seção",
            "Materiais",
            "Resultados",
            "Memorial de Cálculo",
            "Sobre",
        ],
        index=0,
    )

    st.divider()

    if st.button("Limpar carregamentos", use_container_width=True):
        st.session_state.loads = []
        st.session_state.results = None
        st.success("Carregamentos removidos.")

    if st.button("Limpar resultados", use_container_width=True):
        st.session_state.results = None
        st.success("Resultados removidos.")
        # ============================================================
# PÁGINA: PROPRIEDADES DA SEÇÃO
# ============================================================

def render_section_page():
    st.subheader("Propriedades da Seção Transversal")

    sec_mode = st.radio(
        "Modo",
        ["Catálogo interno", "Dimensões (manual)"],
        horizontal=True,
        index=0,
    )

    Iy = Iz = 0.0
    yext = (0.0, 0.0)
    zext = (0.0, 0.0)
    sec_desc = ""

    if sec_mode == "Catálogo interno":
        df = st.session_state.profiles_df.copy()
        fam = st.selectbox("Família", sorted(df["family"].unique()))
        df2 = df[df["family"] == fam]
        name = st.selectbox("Perfil", df2["name"].tolist())
        row = df2[df2["name"] == name].iloc[0].to_dict()

        famU = fam.upper().strip()
        sec_desc = f"{famU} | {name}"

        if famU == "RETANGULO":
            Iy, Iz, yext, zext = rect_Iy_Iz(mm_to_m(row["b_mm"]), mm_to_m(row["h_mm"]))
        elif famU == "BARRA REDONDA":
            Iy, Iz, yext, zext = round_solid_I(mm_to_m(row["d_mm"]))
        elif famU == "TUBO REDONDO":
            Iy, Iz, yext, zext = round_tube_I(mm_to_m(row["od_mm"]), mm_to_m(row["t_mm"]))
        elif famU in ["TUBO QUADRADO", "TUBO RETANGULAR"]:
            Iy, Iz, yext, zext = rect_tube_I(mm_to_m(row["b_mm"]), mm_to_m(row["h_mm"]), mm_to_m(row["t_mm"]))

    else:
        unit = st.selectbox("Unidade", ["mm", "m"])
        b = st.number_input(f"Largura b ({unit})", value=100.0)
        h = st.number_input(f"Altura h ({unit})", value=10.0)

        b_m = mm_to_m(b) if unit == "mm" else b
        h_m = mm_to_m(h) if unit == "mm" else h

        Iy, Iz, yext, zext = rect_Iy_Iz(b_m, h_m)
        sec_desc = f"RETANGULAR | b={b} {unit}, h={h} {unit}"

    c1, c2 = st.columns(2)
    c1.metric("Iy", f"{Iy:.3e} m⁴")
    c2.metric("Iz", f"{Iz:.3e} m⁴")

    if st.button("Aplicar seção", type="primary"):
        st.session_state.Iy = Iy
        st.session_state.Iz = Iz
        st.session_state.yext = yext
        st.session_state.zext = zext
        st.session_state.sec_desc = sec_desc
        st.success("Seção aplicada.")


# ============================================================
# PÁGINA: MATERIAIS
# ============================================================

def render_materials_page():
    st.subheader("Materiais")

    mats = st.session_state.materials

    st.dataframe(
        pd.DataFrame([{ "material": k, **v } for k, v in mats.items()]),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Adicionar / Editar material")

    with st.form("mat_form"):
        name = st.text_input("ID do material", value="novo_material").strip().lower()
        E_MPa = st.number_input("E (MPa)", value=210000.0)
        fy_MPa = st.number_input("fy (MPa)", value=350.0)
        nu = st.number_input("ν", value=0.30)

        save = st.form_submit_button("Salvar", type="primary")

    if save:
        st.session_state.materials[name] = {
            "E": MPa_to_Pa(E_MPa),
            "fy": MPa_to_Pa(fy_MPa),
            "nu": nu,
        }
        st.success("Material salvo.")
        # ============================================================
# PÁGINA: CALCULADORA (modelo + cargas + execução)
# ============================================================

def render_calculadora_page():
    st.subheader("Calculadora")

    # --- parâmetros do modelo (topo)
    unit_system = st.selectbox(
        "Sistema de unidades (entrada)",
        ["mm (mm, N, MPa)", "SI (m, N, Pa)"],
        index=0 if st.session_state.unit_system.startswith("mm") else 1
    )
    st.session_state.unit_system = unit_system
    unit_len = "mm" if unit_system.startswith("mm") else "m"

    L_in = st.number_input(f"Comprimento L ({unit_len})", value=float(st.session_state.L_in))
    st.session_state.L_in = L_in
    L_m = mm_to_m(L_in) if unit_len == "mm" else float(L_in)

    c1, c2, c3 = st.columns(3)
    with c1:
        apoio_esq = st.selectbox("Apoio esquerdo", ["Engastado", "Apoio simples (v=0)", "Livre"],
                                 index=["Engastado", "Apoio simples (v=0)", "Livre"].index(st.session_state.apoio_esq))
        st.session_state.apoio_esq = apoio_esq
    with c2:
        apoio_dir = st.selectbox("Apoio direito", ["Engastado", "Apoio simples (v=0)", "Livre"],
                                 index=["Engastado", "Apoio simples (v=0)", "Livre"].index(st.session_state.apoio_dir))
        st.session_state.apoio_dir = apoio_dir
    with c3:
        material = st.selectbox("Material", sorted(st.session_state.materials.keys()),
                                index=sorted(st.session_state.materials.keys()).index(st.session_state.material)
                                if st.session_state.material in st.session_state.materials else 0)
        st.session_state.material = material

    FS = st.number_input("Fator de segurança (FS)", value=float(st.session_state.FS), min_value=1.0)
    st.session_state.FS = FS

    lim = st.selectbox("Limite de deslocamento", ["L/200", "L/250", "L/300", "L/400"],
                       index=["L/200", "L/250", "L/300", "L/400"].index(st.session_state.lim_flecha))
    st.session_state.lim_flecha = lim
    den = {"L/200": 200, "L/250": 250, "L/300": 300, "L/400": 400}[lim]
    delta_adm_m = L_m / den

    ne = st.slider("Discretização (nº de elementos)", 40, 250, int(st.session_state.ne), 10)
    st.session_state.ne = ne

    st.info(f"Seção ativa: {st.session_state.sec_desc}  |  Iy={st.session_state.Iy:.2e} m⁴  |  Iz={st.session_state.Iz:.2e} m⁴")

    st.divider()

    # ============================================================
    # CADASTRO DE CARGAS
    # ============================================================
    st.subheader("Carregamentos")

    plane = st.selectbox("Plano", ["Plano XY (força em Z)", "Plano XZ (força em Y)"])
    kind = st.selectbox("Tipo", ["Força concentrada", "Momento concentrado", "Distribuída (UDL)"])
    sign = st.selectbox("Sentido", ["+", "-"])

    x_in = st.number_input(f"Posição x ({unit_len})", min_value=0.0, max_value=float(L_in), value=float(L_in)/2)
    x_m = mm_to_m(x_in) if unit_len == "mm" else float(x_in)

    if kind == "Força concentrada":
        P_N = st.number_input("P (N)", value=1000.0)
        M_Nm = 0.0
        a_m = b_m = 0.0
        w = 0.0
    elif kind == "Momento concentrado":
        P_N = 0.0
        M_Nm = st.number_input("M (N·m)", value=100.0)
        a_m = b_m = 0.0
        w = 0.0
    else:
        a_in = st.number_input(f"Início a ({unit_len})", value=0.0)
        b_in = st.number_input(f"Fim b ({unit_len})", value=float(L_in))
        w_in = st.number_input("w (N/m)", value=500.0)
        P_N = 0.0
        M_Nm = 0.0
        a_m = mm_to_m(min(a_in, b_in)) if unit_len == "mm" else float(min(a_in, b_in))
        b_m = mm_to_m(max(a_in, b_in)) if unit_len == "mm" else float(max(a_in, b_in))
        w = float(w_in)

    if st.button("Adicionar carga", type="primary"):
        sgn = 1.0 if sign == "+" else -1.0
        st.session_state.loads.append({
            "plane": plane,
            "kind": kind,
            "x_m": float(np.clip(x_m, 0.0, L_m)),
            "P": float(P_N) * sgn if kind == "Força concentrada" else 0.0,
            "M": float(M_Nm) * sgn if kind == "Momento concentrado" else 0.0,
            "a_m": float(a_m) if kind == "Distribuída (UDL)" else 0.0,
            "b_m": float(b_m) if kind == "Distribuída (UDL)" else 0.0,
            "w": float(w) * sgn if kind == "Distribuída (UDL)" else 0.0,
        })
        st.success("Carga adicionada.")

    if st.session_state.loads:
        st.dataframe(pd.DataFrame(st.session_state.loads), use_container_width=True, hide_index=True)
        st.plotly_chart(plot_preview_model(L_m, st.session_state.loads), use_container_width=True)
    else:
        st.info("Nenhuma carga cadastrada.")

    st.divider()

    # ============================================================
    # EXECUÇÃO
    # ============================================================
    if st.button("Executar análise", type="primary", use_container_width=True):

        if st.session_state.Iy <= 0 or st.session_state.Iz <= 0:
            st.error("Seção inválida. Vá em 'Propriedades da Seção' e aplique uma seção válida.")
            st.stop()

        if len(st.session_state.loads) == 0:
            st.error("Cadastre pelo menos 1 carga.")
            st.stop()

        mat = st.session_state.materials[st.session_state.material]
        E = float(mat["E"])
        fy = float(mat["fy"])
        Iy = float(st.session_state.Iy)
        Iz = float(st.session_state.Iz)

        loads_z = []
        loads_y = []

        for ld in st.session_state.loads:
            if "XY" in ld["plane"]:
                if ld["kind"] == "Força concentrada":
                    loads_z.append({"type": "P", "x": ld["x_m"], "P": ld["P"]})
                elif ld["kind"] == "Momento concentrado":
                    loads_z.append({"type": "M", "x": ld["x_m"], "M": ld["M"]})
                else:
                    loads_z.append({"type": "w", "a": ld["a_m"], "b": ld["b_m"], "w": ld["w"]})
            else:
                if ld["kind"] == "Força concentrada":
                    loads_y.append({"type": "P", "x": ld["x_m"], "P": ld["P"]})
                elif ld["kind"] == "Momento concentrado":
                    loads_y.append({"type": "M", "x": ld["x_m"], "M": ld["M"]})
                else:
                    loads_y.append({"type": "w", "a": ld["a_m"], "b": ld["b_m"], "w": ld["w"]})

        # Solver Z (usa Iy)
        xs = np.linspace(0.0, L_m, ne+1)
        wz = np.zeros_like(xs)
        Vz = np.zeros_like(xs)
        My = np.zeros_like(xs)
        reac_z = {}
        if loads_z:
            xs, wz, Vz, My, reac_z = solve_beam_FEM(L_m, E*Iy, apoio_esq, apoio_dir, loads_z, ne=ne)

        # Solver Y (usa Iz)
        xs2 = np.linspace(0.0, L_m, ne+1)
        wy = np.zeros_like(xs2)
        Vy = np.zeros_like(xs2)
        Mz = np.zeros_like(xs2)
        reac_y = {}
        if loads_y:
            xs2, wy, Vy, Mz, reac_y = solve_beam_FEM(L_m, E*Iz, apoio_esq, apoio_dir, loads_y, ne=ne)

        # Resultante
        if len(xs2) == len(xs):
            w_res = np.sqrt(wy**2 + wz**2)
        else:
            w_res = np.sqrt(np.interp(xs, xs2, wy)**2 + wz**2)

        delta_max_m = float(np.max(np.abs(w_res)))
        idx_max = int(np.argmax(np.abs(w_res)))
        x_at_max = float(xs[idx_max])

        My_max = float(np.max(np.abs(My))) if len(My) else 0.0
        Mz_max = float(np.max(np.abs(Mz))) if len(Mz) else 0.0

        ymin, ymax = st.session_state.yext
        zmin, zmax = st.session_state.zext
        corners = [(ymin, zmin), (ymin, zmax), (ymax, zmin), (ymax, zmax)]

        def sigma_at(y, z, My_, Mz_):
            return (My_ * z) / Iy + (Mz_ * y) / Iz

        sigma_max = float(np.max(np.abs([sigma_at(y, z, My_max, Mz_max) for (y, z) in corners])))
        sigma_vm = abs(sigma_max)
        sigma_adm = fy / FS

        ok_defl = delta_max_m <= delta_adm_m
        ok_sigma = sigma_vm <= sigma_adm
        ok_yield = sigma_vm <= fy

        st.session_state.results = {
            "meta": {
                "L_m": L_m,
                "apoio_esq": apoio_esq,
                "apoio_dir": apoio_dir,
                "FS": FS,
                "lim_flecha": lim,
                "delta_adm_m": delta_adm_m,
                "sec_desc": st.session_state.sec_desc,
                "material": st.session_state.material,
                "E": E,
                "fy": fy,
                "Iy": Iy,
                "Iz": Iz,
            },
            "series": {
                "x_m": xs.tolist(),
                "wz_m": wz.tolist(),
                "wy_m": wy.tolist() if len(wy) == len(xs) else np.interp(xs, xs2, wy).tolist(),
                "wres_m": w_res.tolist(),
                "My_Nm": My.tolist(),
                "Mz_Nm": Mz.tolist() if len(Mz) == len(xs) else np.interp(xs, xs2, Mz).tolist(),
                "Vz_N": Vz.tolist(),
                "Vy_N": Vy.tolist() if len(Vy) == len(xs) else np.interp(xs, xs2, Vy).tolist(),
            },
            "peaks": {
                "delta_max_m": delta_max_m,
                "x_delta_m": x_at_max,
                "sigma_vm_Pa": sigma_vm,
                "sigma_adm_Pa": sigma_adm,
                "ok_defl": ok_defl,
                "ok_sigma": ok_sigma,
                "ok_yield": ok_yield,
            },
            "reactions": {
                "reac_z": reac_z,
                "reac_y": reac_y,
            }
        }

        st.success("Análise concluída. Agora vá na página **Resultados**.")
        # ============================================================
# PÁGINA: RESULTADOS (Plotly)
# ============================================================

def render_results_page():
    st.subheader("Resultados (Plotly)")

    if st.session_state.results is None:
        st.warning("Execute a análise na página **Calculadora** primeiro.")
        return

    res = st.session_state.results
    meta = res["meta"]
    ser = res["series"]
    pk = res["peaks"]
    reac = res["reactions"]

    delta_max_mm = m_to_mm(pk["delta_max_m"])
    delta_adm_mm = m_to_mm(meta["delta_adm_m"])

    veredito = "ATENDE ✅" if (pk["ok_defl"] and pk["ok_sigma"] and pk["ok_yield"]) else "NÃO ATENDE ❌"
    ver_defl = "OK ✅" if pk["ok_defl"] else "NÃO OK ❌"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("δmax (mm)", f"{delta_max_mm:.4f}")
        st.caption(f"Limite {meta['lim_flecha']}: {delta_adm_mm:.4f} mm")
    with c2:
        st.metric("x no pico (m)", f"{pk['x_delta_m']:.3f}")
    with c3:
        st.metric("Status deslocamento", ver_defl)
    with c4:
        st.metric("Veredito global", veredito)

    st.divider()

    tab_disp, tab_mom, tab_shear, tab_reac, tab_check = st.tabs(
        ["Deslocamentos", "Momentos", "Cortantes", "Reações", "Verificações"]
    )

    x = ser["x_m"]

    # --- Deslocamentos
    with tab_disp:
        wres_mm = [m_to_mm(v) for v in ser["wres_m"]]
        wy_mm   = [m_to_mm(v) for v in ser["wy_m"]]
        wz_mm   = [m_to_mm(v) for v in ser["wz_m"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=wres_mm, mode="lines", name="w_resultante (mm)"))
        fig.add_trace(go.Scatter(x=x, y=wy_mm,   mode="lines", name="wy (mm)"))
        fig.add_trace(go.Scatter(x=x, y=wz_mm,   mode="lines", name="wz (mm)"))
        fig.add_trace(go.Scatter(x=[pk["x_delta_m"]], y=[delta_max_mm], mode="markers", name="δmax"))
        fig.update_layout(xaxis_title="x (m)", yaxis_title="w (mm)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # --- Momentos
    with tab_mom:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ser["My_Nm"], mode="lines", name="My (XY → Fz)"))
        fig.add_trace(go.Scatter(x=x, y=ser["Mz_Nm"], mode="lines", name="Mz (XZ → Fy)"))
        fig.update_layout(xaxis_title="x (m)", yaxis_title="M (N·m)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # --- Cortantes
    with tab_shear:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ser["Vz_N"], mode="lines", name="Vz (XY → Fz)"))
        fig.add_trace(go.Scatter(x=x, y=ser["Vy_N"], mode="lines", name="Vy (XZ → Fy)"))
        fig.update_layout(xaxis_title="x (m)", yaxis_title="V (N)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # --- Reações
    with tab_reac:
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Plano XY → forças em Z**")
            if reac["reac_z"]:
                st.dataframe(reactions_table(reac["reac_z"]), use_container_width=True, hide_index=True)
            else:
                st.info("Sem cargas no plano XY.")
        with cR:
            st.markdown("**Plano XZ → forças em Y**")
            if reac["reac_y"]:
                st.dataframe(reactions_table(reac["reac_y"]), use_container_width=True, hide_index=True)
            else:
                st.info("Sem cargas no plano XZ.")

    # --- Verificações
    with tab_check:
        sigma_vm_MPa = pk["sigma_vm_Pa"] / 1e6
        sigma_adm_MPa = pk["sigma_adm_Pa"] / 1e6
        fy_MPa = meta["fy"] / 1e6

        c1, c2, c3 = st.columns(3)
        c1.metric("σvm,max (MPa)", f"{sigma_vm_MPa:.2f}")
        c2.metric("σadm = fy/FS (MPa)", f"{sigma_adm_MPa:.2f}")
        c3.metric("fy (MPa)", f"{fy_MPa:.2f}")

        st.write(f"σvm ≤ σadm: {'OK ✅' if pk['ok_sigma'] else 'NÃO OK ❌'}")
        st.write(f"σvm ≤ fy: {'OK ✅' if pk['ok_yield'] else 'NÃO OK ❌'}")


# ============================================================
# PÁGINA: MEMORIAL
# ============================================================

def render_memorial_page():
    st.subheader("Memorial de Cálculo (síntese)")

    st.markdown(
        r"""
**Modelo**
- Viga de Euler–Bernoulli (pequenas deformações).
- Flexão em dois planos independentes (XY→Fz e XZ→Fy).
- MEF 1D com 2 GDL por nó: deslocamento transversal \(v\) e rotação \(\theta\).

**Tensão (flexão biaxial)**
\[
\sigma(y,z) = \frac{M_y z}{I_y} + \frac{M_z y}{I_z}
\]

**Critérios**
- Deslocamento admissível: \(\delta_{adm} = L/n\) com \(n\in\{200,250,300,400\}\).
- Tensão admissível: \(\sigma_{adm} = f_y/FS\).
- Escoamento: \(\sigma_{vm} \le f_y\) (flexão pura, \(\tau\approx 0\)).

**Limitações**
- Torção não calculada.
- Não considera Timoshenko (cisalhamento).
- Não considera flambagem/instabilidade global.
"""
    )


# ============================================================
# PÁGINA: SOBRE
# ============================================================

def render_about_page():
    st.subheader("Sobre")
    st.markdown(
        """
**App:** Análise de Flexão em Vigas (Biaxial)  
**Stack:** Python • Streamlit • Plotly  
**Foco:** cálculo + visualização (deslocamento primeiro)

Evoluções recomendadas:
- Torção (GJ) e rotação θx
- Catálogo robusto de perfis (CSV/Excel validado)
- Exportar relatório (PDF/HTML)
"""
    )


# ============================================================
# ROUTER FINAL (OBRIGATÓRIO)
# ============================================================

if page == "Calculadora":
    render_calculadora_page()
elif page == "Propriedades da Seção":
    render_section_page()
elif page == "Materiais":
    render_materials_page()
elif page == "Resultados":
    render_results_page()
elif page == "Memorial de Cálculo":
    render_memorial_page()
else:
    render_about_page()
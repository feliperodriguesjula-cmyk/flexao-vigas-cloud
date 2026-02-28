# app.py  —  Calculadora de Flexão em Vigas (Biaxial) | Streamlit Single-File "Multipage"
# Autor: (seu nome)
# Objetivo: UI profissional + organização em páginas internas (sidebar) + gráficos Plotly
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
# AUTH (opcional, mas recomendado em produção)
# ============================================================
def require_auth():
    """
    Usa st.secrets["APP_PASSWORD"].
    - No Streamlit Cloud: defina em App -> Settings -> Secrets
    - Local: você pode criar .streamlit/secrets.toml
      APP_PASSWORD="sua_senha"
    """
    # Se não existir segredo, não bloqueia (facilita dev local).
    # Em produção, coloque APP_PASSWORD nos Secrets.
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
# UTIL: UNIDADES (motor SI)
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
    {"family":"RETANGULO",        "name":"Ret 100 x 10",         "b_mm":100.0, "h_mm":10.0},
    {"family":"BARRA REDONDA",    "name":"Barra Ø20",            "d_mm":20.0},
    {"family":"TUBO REDONDO",     "name":"Tubo Ø60,3 x 3,0",     "od_mm":60.3, "t_mm":3.0},
    {"family":"TUBO QUADRADO",    "name":"Tubo 50x50x3,0",       "b_mm":50.0,  "h_mm":50.0, "t_mm":3.0},
    {"family":"TUBO RETANGULAR",  "name":"Tubo 80x40x3,0",       "b_mm":80.0,  "h_mm":40.0, "t_mm":3.0},
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

# Inicializa seção default (se ainda não calculada)
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

    # pós: montar V e M por elemento (forças internas nos nós do elemento)
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
# HELPERS: reações e plot preview
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
    st.caption("Modelo Euler–Bernoulli • MEF 1D • Saída principal em mm • Gráficos interativos (Plotly)")

st.divider()

# ============================================================
# NAVEGAÇÃO (Multipage dentro de 1 arquivo)
# ============================================================
with st.sidebar:
    st.header("Navegação")
    page = st.radio(
        "Página",
        ["Calculadora", "Propriedades da Seção", "Materiais", "Resultados", "Memorial de Cálculo", "Sobre"],
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
    st.caption("Defina a seção para cálculo de Iy e Iz (m⁴).")

    sec_mode = st.radio(
        "Modo",
        ["Catálogo interno", "Dimensões (manual)", "Importar Excel/CSV (catálogo)"],
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

        famU = str(fam).upper().strip()
        sec_desc = f"{famU} | {name}"

        if famU == "RETANGULO":
            Iy, Iz, yext, zext = rect_Iy_Iz(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])))
        elif famU == "BARRA REDONDA":
            Iy, Iz, yext, zext = round_solid_I(mm_to_m(float(row["d_mm"])))
        elif famU == "TUBO REDONDO":
            Iy, Iz, yext, zext = round_tube_I(mm_to_m(float(row["od_mm"])), mm_to_m(float(row["t_mm"])))
        elif famU in ["TUBO QUADRADO", "TUBO RETANGULAR"]:
            Iy, Iz, yext, zext = rect_tube_I(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])), mm_to_m(float(row["t_mm"])))
        else:
            st.error("Família não suportada no catálogo interno.")

    elif sec_mode == "Dimensões (manual)":
        fam = st.selectbox("Tipo de seção", ["RETANGULAR", "CIRCULAR MACIÇA", "TUBO CIRCULAR", "TUBO RETANGULAR/QUADRADO"], index=0)
        unit_len = st.selectbox("Unidade de entrada", ["mm", "m"], index=0)

        if fam == "RETANGULAR":
            b = st.number_input(f"Largura b ({unit_len})", min_value=0.001, value=100.0 if unit_len=="mm" else 0.10)
            h = st.number_input(f"Altura h ({unit_len})",  min_value=0.001, value=10.0  if unit_len=="mm" else 0.010)
            b_m = mm_to_m(b) if unit_len=="mm" else float(b)
            h_m = mm_to_m(h) if unit_len=="mm" else float(h)
            Iy, Iz, yext, zext = rect_Iy_Iz(b_m, h_m)
            sec_desc = f"RETANGULAR | b={b} {unit_len}, h={h} {unit_len}"

        elif fam == "CIRCULAR MACIÇA":
            d = st.number_input(f"Diâmetro d ({unit_len})", min_value=0.001, value=20.0 if unit_len=="mm" else 0.02)
            d_m = mm_to_m(d) if unit_len=="mm" else float(d)
            Iy, Iz, yext, zext = round_solid_I(d_m)
            sec_desc = f"CIRCULAR MACIÇA | d={d} {unit_len}"

        elif fam == "TUBO CIRCULAR":
            od = st.number_input(f"Diâmetro externo OD ({unit_len})", min_value=0.001, value=60.3 if unit_len=="mm" else 0.0603)
            t  = st.number_input(f"Espessura t ({unit_len})",        min_value=0.0005, value=3.0 if unit_len=="mm" else 0.003)
            od_m = mm_to_m(od) if unit_len=="mm" else float(od)
            t_m  = mm_to_m(t)  if unit_len=="mm" else float(t)
            Iy, Iz, yext, zext = round_tube_I(od_m, t_m)
            sec_desc = f"TUBO CIRCULAR | OD={od} {unit_len}, t={t} {unit_len}"

        else:
            b = st.number_input(f"Largura B ({unit_len})", min_value=0.001, value=80.0 if unit_len=="mm" else 0.08)
            h = st.number_input(f"Altura H ({unit_len})",  min_value=0.001, value=40.0 if unit_len=="mm" else 0.04)
            t = st.number_input(f"Espessura t ({unit_len})", min_value=0.0005, value=3.0 if unit_len=="mm" else 0.003)
            b_m = mm_to_m(b) if unit_len=="mm" else float(b)
            h_m = mm_to_m(h) if unit_len=="mm" else float(h)
            t_m = mm_to_m(t) if unit_len=="mm" else float(t)
            Iy, Iz, yext, zext = rect_tube_I(b_m, h_m, t_m)
            sec_desc = f"TUBO RET/QUAD | B={b} {unit_len}, H={h} {unit_len}, t={t} {unit_len}"

    else:
        st.caption("Importe um catálogo Excel/CSV com colunas: family,name,b_mm,h_mm,t_mm,d_mm,od_mm")
        up = st.file_uploader("Enviar Excel/CSV", type=["xlsx", "xls", "csv"])
        if up is not None:
            if up.name.lower().endswith(".csv"):
                cat = pd.read_csv(up)
            else:
                cat = pd.read_excel(up)

            st.dataframe(cat, use_container_width=True, hide_index=True)
            fam = st.selectbox("Família", sorted(cat["family"].unique()))
            cat2 = cat[cat["family"] == fam]
            name = st.selectbox("Perfil", cat2["name"].tolist())
            row = cat2[cat2["name"] == name].iloc[0].to_dict()

            famU = str(fam).upper().strip()
            sec_desc = f"{famU} | {name}"

            if famU == "RETANGULO":
                Iy, Iz, yext, zext = rect_Iy_Iz(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])))
            elif famU == "BARRA REDONDA":
                Iy, Iz, yext, zext = round_solid_I(mm_to_m(float(row["d_mm"])))
            elif famU == "TUBO REDONDO":
                Iy, Iz, yext, zext = round_tube_I(mm_to_m(float(row["od_mm"])), mm_to_m(float(row["t_mm"])))
            elif famU in ["TUBO QUADRADO", "TUBO RETANGULAR"]:
                Iy, Iz, yext, zext = rect_tube_I(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])), mm_to_m(float(row["t_mm"])))
            else:
                st.error("Família não suportada no import (ainda).")

    cA, cB, cC = st.columns([1.3, 1.0, 1.0])
    with cA:
        st.write(f"**Seção ativa:** {sec_desc if sec_desc else st.session_state.sec_desc}")
    with cB:
        st.write(f"**Iy:** {Iy if Iy>0 else st.session_state.Iy:.3e} m⁴")
    with cC:
        st.write(f"**Iz:** {Iz if Iz>0 else st.session_state.Iz:.3e} m⁴")

    if st.button("Aplicar seção", type="primary"):
        if Iy <= 0 or Iz <= 0:
            st.error("Seção inválida (Iy/Iz ≤ 0). Verifique dimensões.")
            return
        st.session_state.Iy = float(Iy)
        st.session_state.Iz = float(Iz)
        st.session_state.yext = yext
        st.session_state.zext = zext
        st.session_state.sec_desc = sec_desc if sec_desc else st.session_state.sec_desc
        st.success("Seção aplicada ao modelo.")

# ============================================================
# PÁGINA: MATERIAIS
# ============================================================
def render_materials_page():
    st.subheader("Materiais")
    st.caption("Gerencie propriedades mecânicas: módulo de elasticidade (E), escoamento (fy) e Poisson (ν).")

    mats = st.session_state.materials
    st.dataframe(
        pd.DataFrame([{ "material": k, **v } for k, v in mats.items()]),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Adicionar / Editar material")
    with st.form("mat_form"):
        name = st.text_input("ID do material (ex.: aco_sae_1020)", value="meu_material").strip().lower()
        unit = st.selectbox("Unidade de E e fy", ["MPa", "Pa"], index=0)
        E_in  = st.number_input(f"E ({unit})", value=210000.0 if unit=="MPa" else 210e9)
        fy_in = st.number_input(f"fy ({unit})", value=350.0    if unit=="MPa" else 350e6)
        nu_in = st.number_input("ν (Poisson)", value=0.30, min_value=0.0, max_value=0.49)

        save = st.form_submit_button("Salvar", type="primary")

    if save:
        if not name:
            st.error("Informe um ID válido.")
            return
        E_val  = MPa_to_Pa(E_in)  if unit=="MPa" else float(E_in)
        fy_val = MPa_to_Pa(fy_in) if unit=="MPa" else float(fy_in)
        st.session_state.materials[name] = {"E": float(E_val), "fy": float(fy_val), "nu": float(nu_in)}
        st.success("Material salvo.")

# ============================================================
# PÁGINA: CALCULADORA (modelo + cargas + execução)
# ============================================================
def render_calculadora_page():
    st.subheader("Calculadora")
    st.caption("Defina o modelo estrutural, cadastre carregamentos e execute a análise.")

    # --- Sidebar: parâmetros principais (form)
    with st.sidebar:
        st.header("Parâmetros do modelo")
        with st.form("model_form"):
            st.session_state.unit_system = st.selectbox(
                "Sistema de unidades (entrada)",
                ["mm (mm, N, MPa)", "SI (m, N, Pa)"],
                index=0 if st.session_state.unit_system.startswith("mm") else 1
            )
            unit_system = st.session_state.unit_system
            unit_len = "mm" if unit_system.startswith("mm") else "m"

            st.session_state.L_in = st.number_input(
                f"Comprimento L ({unit_len})",
                min_value=0.001,
                value=float(st.session_state.L_in),
                step=50.0 if unit_len=="mm" else 0.1
            )

            st.session_state.apoio_esq = st.selectbox(
                "Condição de contorno (esquerda)",
                ["Engastado", "Apoio simples (v=0)", "Livre"],
                index=["Engastado", "Apoio simples (v=0)", "Livre"].index(st.session_state.apoio_esq)
            )

            st.session_state.apoio_dir = st.selectbox(
                "Condição de contorno (direita)",
                ["Engastado", "Apoio simples (v=0)", "Livre"],
                index=["Engastado", "Apoio simples (v=0)", "Livre"].index(st.session_state.apoio_dir)
            )

            st.session_state.material = st.selectbox(
                "Material",
                sorted(list(st.session_state.materials.keys())),
                index=sorted(list(st.session_state.materials.keys())).index(st.session_state.material)
                if st.session_state.material in st.session_state.materials else 0
            )

            st.session_state.FS = st.number_input("Fator de segurança (FS)", min_value=1.0, value=float(st.session_state.FS), step=0.1)
            st.session_state.lim_flecha = st.selectbox("Limite de deslocamento", ["L/200","L/250","L/300","L/400"], index=["L/200","L/250","L/300","L/400"].index(st.session_state.lim_flecha))
            st.session_state.ne = st.slider("Discretização (nº de elementos)", 40, 250, int(st.session_state.ne), 10)

            applied = st.form_submit_button("Aplicar", type="primary")

    # --- Resumo do modelo (topo)
    unit_system = st.session_state.unit_system
    unit_len = "mm" if unit_system.startswith("mm") else "m"
    L_m = mm_to_m(st.session_state.L_in) if unit_len=="mm" else float(st.session_state.L_in)

    mat = st.session_state.materials[st.session_state.material]
    E = mat["E"]
    fy = mat["fy"]
    FS = float(st.session_state.FS)
    lim = st.session_state.lim_flecha
    den = {"L/200":200,"L/250":250,"L/300":300,"L/400":400}[lim]
    delta_adm_m = L_m / den

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comprimento", f"{L_m:.3f} m")
    c2.metric("Seção ativa", st.session_state.sec_desc if st.session_state.sec_desc else "—")
    c3.metric("Iy / Iz", f"{st.session_state.Iy:.2e} / {st.session_state.Iz:.2e} m⁴")
    c4.metric("Limite δadm", f"{m_to_mm(delta_adm_m):.3f} mm")

    st.divider()

    # --- Tabs: Carregamentos | Preview | Validações
    tab_load, tab_prev, tab_val = st.tabs(["Carregamentos", "Pré-visualização", "Validações"])

    plane_map = {
        "Plano XY (força em Z)": "Z",  # usa Iy
        "Plano XZ (força em Y)": "Y",  # usa Iz
    }

    with tab_load:
        st.subheader("Cadastro de carregamentos")
        st.caption("Flexão tratada em dois planos independentes. Carregamento fora do centro induz torção (não contabilizada).")

        with st.form("load_form"):
            cA, cB, cC = st.columns([1.6, 1.2, 1.0])
            plane = cA.selectbox("Plano de aplicação", list(plane_map.keys()))
            kind  = cB.selectbox("Tipo", ["Força concentrada", "Momento concentrado", "Carga distribuída (UDL)"])
            sign  = cC.selectbox("Sentido", ["+", "-"])

            x_in = st.number_input(
                f"Posição x ({unit_len})",
                min_value=0.0, max_value=float(st.session_state.L_in),
                value=float(st.session_state.L_in)/2
            )
            coord_in = st.number_input("Coordenada no plano (referência)", value=0.0)
            aplicar_no_centro = st.checkbox("Aplicar no centro geométrico (ignorar torção)", value=True)

            if kind == "Força concentrada":
                P_N = st.number_input("Magnitude P (N)", value=1000.0, step=100.0)
                M_Nm = 0.0
                a_in = b_in = 0.0
                w_Nm = 0.0
            elif kind == "Momento concentrado":
                P_N = 0.0
                M_Nm = st.number_input("Magnitude M (N·m)", value=100.0, step=10.0)
                a_in = b_in = 0.0
                w_Nm = 0.0
            else:
                c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
                a_in = c1.number_input(f"Início a ({unit_len})", value=0.0)
                b_in = c2.number_input(f"Fim b ({unit_len})", value=float(st.session_state.L_in))
                w_Nm = c3.number_input("Intensidade w (N/m)", value=500.0, step=50.0)
                P_N = 0.0
                M_Nm = 0.0

            add = st.form_submit_button("Adicionar", type="primary")

        if add:
            x_m = mm_to_m(x_in) if unit_len=="mm" else float(x_in)
            sgn = 1.0 if sign == "+" else -1.0
            P = float(P_N) * sgn
            M = float(M_Nm) * sgn
            a_m = mm_to_m(min(a_in, b_in)) if unit_len=="mm" else float(min(a_in, b_in))
            b_m = mm_to_m(max(a_in, b_in)) if unit_len=="mm" else float(max(a_in, b_in))
            w = float(w_Nm) * sgn

            st.session_state.loads.append({
                "plane": plane,
                "kind": kind,
                "sign": sign,
                "x_m": float(np.clip(x_m, 0.0, L_m)),
                "coord_ref": float(coord_in),
                "centered": bool(aplicar_no_centro),
                "P": P if kind=="Força concentrada" else 0.0,
                "M": M if kind=="Momento concentrado" else 0.0,
                "a_m": a_m if kind.startswith("Carga distribuída") else 0.0,
                "b_m": b_m if kind.startswith("Carga distribuída") else 0.0,
                "w": w if kind.startswith("Carga distribuída") else 0.0,
            })
            st.success("Carregamento adicionado.")

        if st.session_state.loads:
            df = pd.DataFrame(st.session_state.loads)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum carregamento cadastrado.")

    with tab_prev:
        st.subheader("Pré-visualização do modelo")
        st.plotly_chart(plot_preview_model(L_m, st.session_state.loads), use_container_width=True)

    with tab_val:
        st.subheader("Validações e limitações")
        if st.session_state.Iy <= 0 or st.session_state.Iz <= 0:
            st.error("Seção inválida (Iy/Iz ≤ 0). Configure a seção em **Propriedades da Seção**.")
        else:
            st.success("Seção válida (Iy e Iz > 0).")

        st.warning("Modelo Euler–Bernoulli: não considera deformação por cisalhamento (Timoshenko).")
        st.warning("Torção não é calculada. Cargas fora do centro geram torção.")
        st.info("Para evolução futura: incluir módulo de torção (G·J) e rotação θx.")

    # --- EXECUÇÃO
    st.divider()
    run = st.button("Executar análise", type="primary", use_container_width=True)

    if run:
        if st.session_state.Iy <= 0 or st.session_state.Iz <= 0:
            st.error("Seção inválida. Ajuste em **Propriedades da Seção**.")
            st.stop()
        if len(st.session_state.loads) == 0:
            st.error("Cadastre ao menos um carregamento antes de executar.")
            st.stop()

        Iy = float(st.session_state.Iy)
        Iz = float(st.session_state.Iz)
        apoio_esq = st.session_state.apoio_esq
        apoio_dir = st.session_state.apoio_dir
        ne = int(st.session_state.ne)

        loads_z = []  # plano XY -> força em Z -> usa Iy (momento My)
        loads_y = []  # plano XZ -> força em Y -> usa Iz (momento Mz)

        for ld in st.session_state.loads:
            if "XY" in ld["plane"]:
                if ld["kind"] == "Força concentrada":
                    loads_z.append({"type":"P", "x": ld["x_m"], "P": ld["P"]})
                elif ld["kind"] == "Momento concentrado":
                    loads_z.append({"type":"M", "x": ld["x_m"], "M": ld["M"]})
                else:
                    loads_z.append({"type":"w", "a": ld["a_m"], "b": ld["b_m"], "w": ld["w"]})
            else:
                if ld["kind"] == "Força concentrada":
                    loads_y.append({"type":"P", "x": ld["x_m"], "P": ld["P"]})
                elif ld["kind"] == "Momento concentrado":
                    loads_y.append({"type":"M", "x": ld["x_m"], "M": ld["M"]})
                else:
                    loads_y.append({"type":"w", "a": ld["a_m"], "b": ld["b_m"], "w": ld["w"]})

        with st.status("Executando MEF...", expanded=False):
            # Resolver plano XY (Z)
            xs = np.linspace(0.0, L_m, ne+1)
            wz = np.zeros_like(xs)
            Vz = np.zeros_like(xs)
            My = np.zeros_like(xs)
            reac_z = {}
            if len(loads_z) > 0:
                xs, wz, Vz, My, reac_z = solve_beam_FEM(L_m, E*Iy, apoio_esq, apoio_dir, loads_z, ne=ne)

            # Resolver plano XZ (Y)
            xs2 = np.linspace(0.0, L_m, ne+1)
            wy = np.zeros_like(xs2)
            Vy = np.zeros_like(xs2)
            Mz = np.zeros_like(xs2)
            reac_y = {}
            if len(loads_y) > 0:
                xs2, wy, Vy, Mz, reac_y = solve_beam_FEM(L_m, E*Iz, apoio_esq, apoio_dir, loads_y, ne=ne)

            # Resultante (interp se necessário)
            if len(xs2) == len(xs):
                w_res = np.sqrt(wy**2 + wz**2)
            else:
                w_res = np.sqrt(np.interp(xs, xs2, wy)**2 + wz**2)

            delta_max_m = float(np.max(np.abs(w_res)))
            idx_max = int(np.argmax(np.abs(w_res)))
            x_at_max = float(xs[idx_max])

            # picos
            My_max = float(np.max(np.abs(My))) if len(My) else 0.0
            Mz_max = float(np.max(np.abs(Mz))) if len(Mz) else 0.0
            Vz_max = float(np.max(np.abs(Vz))) if len(Vz) else 0.0
            Vy_max = float(np.max(np.abs(Vy))) if len(Vy) else 0.0
            Vmax = max(Vz_max, Vy_max)

            if Vz_max >= Vy_max and len(Vz):
                iV = int(np.argmax(np.abs(Vz))); xV = float(xs[iV])
            elif len(Vy):
                iV = int(np.argmax(np.abs(Vy))); xV = float(xs2[iV])
            else:
                xV = 0.0

            # Tensões (flexão biaxial) – canto da seção (aprox)
            ymin, ymax = st.session_state.yext
            zmin, zmax = st.session_state.zext
            corners = [(ymin,zmin),(ymin,zmax),(ymax,zmin),(ymax,zmax)]

            def sigma_at(y, z, My_, Mz_):
                return (My_ * z) / Iy + (Mz_ * y) / Iz

            sigma_max = float(np.max(np.abs([sigma_at(y,z,My_max,Mz_max) for (y,z) in corners])))
            sigma_vm = abs(sigma_max)  # flexão pura -> tau=0
            sigma_adm = fy / FS

            ok_defl = delta_max_m <= delta_adm_m
            ok_sigma_adm = sigma_vm <= sigma_adm
            ok_yield = sigma_vm <= fy

            # Salvar resultados para a página Resultados
            st.session_state.results = {
                "meta": {
                    "L_m": L_m,
                    "apoio_esq": apoio_esq,
                    "apoio_dir": apoio_dir,
                    "FS": FS,
                    "lim_flecha": lim,
                    "delta_adm_m": delta_adm_m,
                    "sec_desc": st.session_state.sec_desc,
                    "Iy": Iy, "Iz": Iz,
                    "material": st.session_state.material,
                    "E": E, "fy": fy,
                },
                "series": {
                    "x_m": xs.tolist(),
                    "wz_m": wz.tolist(),
                    "wy_m": wy.tolist() if len(wy)==len(xs) else np.interp(xs, xs2, wy).tolist(),
                    "wres_m": w_res.tolist(),
                    "My_Nm": My.tolist(),
                    "Mz_Nm": Mz.tolist() if len(Mz)==len(xs) else np.interp(xs, xs2, Mz).tolist(),
                    "Vz_N": Vz.tolist(),
                    "Vy_N": Vy.tolist() if len(Vy)==len(xs) else np.interp(xs, xs2, Vy).tolist(),
                },
                "peaks": {
                    "delta_max_m": float(delta_max_m),
                    "x_delta_m": float(x_at_max),
                    "My_max_Nm": float(My_max),
                    "Mz_max_Nm": float(Mz_max),
                    "Vmax_N": float(Vmax),
                    "xV_m": float(xV),
                    "sigma_vm_max_Pa": float(sigma_vm),
                    "sigma_adm_Pa": float(sigma_adm),
                    "ok_defl": bool(ok_defl),
                    "ok_sigma_adm": bool(ok_sigma_adm),
                    "ok_yield": bool(ok_yield),
                },
                "reactions": {
                    "reac_z": reac_z,
                    "reac_y": reac_y,
                }
            }

        st.success("Análise concluída. Abra a página **Resultados** para visualizar os gráficos e verificações.")

# ============================================================
# PÁGINA: RESULTADOS (Plotly, deslocamento primeiro)
# ============================================================
def render_results_page():
    st.subheader("Resultados")
    st.caption("Prioridade: deslocamentos (mm) • Gráficos interativos (Plotly)")

    if st.session_state.results is None:
        st.warning("Execute a análise na página **Calculadora**.")
        return

    res = st.session_state.results
    meta = res["meta"]
    ser = res["series"]
    pk = res["peaks"]
    reac = res["reactions"]

    delta_max_mm = m_to_mm(pk["delta_max_m"])
    delta_adm_mm = m_to_mm(meta["delta_adm_m"])

    veredito = "ATENDE ✅" if (pk["ok_defl"] and pk["ok_sigma_adm"] and pk["ok_yield"]) else "NÃO ATENDE ❌"
    ver_defl = "OK ✅" if pk["ok_defl"] else "NÃO OK ❌"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("δmax (deslocamento máximo)", f"{delta_max_mm:.4f} mm")
        st.caption(f"Limite {meta['lim_flecha']}: {delta_adm_mm:.4f} mm")
    with c2:
        st.metric("Posição do pico", f"x = {pk['x_delta_m']:.3f} m")
    with c3:
        st.metric("Status deslocamento", ver_defl)
    with c4:
        st.metric("Veredito global", veredito)

    st.divider()

    tab_disp, tab_mom, tab_shear, tab_reac, tab_check = st.tabs(
        ["Deslocamentos", "Momentos", "Cortantes", "Reações", "Verificações"]
    )

    x = ser["x_m"]

    with tab_disp:
        wres_mm = [m_to_mm(v) for v in ser["wres_m"]]
        wy_mm   = [m_to_mm(v) for v in ser["wy_m"]]
        wz_mm   = [m_to_mm(v) for v in ser["wz_m"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=wres_mm, mode="lines", name="w_resultante (mm)"))
        fig.add_trace(go.Scatter(x=x, y=wy_mm,   mode="lines", name="wy (mm)"))
        fig.add_trace(go.Scatter(x=x, y=wz_mm,   mode="lines", name="wz (mm)"))
        fig.add_trace(go.Scatter(x=[pk["x_delta_m"]], y=[m_to_mm(pk["delta_max_m"])], mode="markers", name="δmax"))
        fig.update_layout(
            xaxis_title="x (m)", yaxis_title="w (mm)",
            height=440, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_mom:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ser["My_Nm"], mode="lines", name="My (plano XY → Fz)"))
        fig.add_trace(go.Scatter(x=x, y=ser["Mz_Nm"], mode="lines", name="Mz (plano XZ → Fy)"))
        fig.update_layout(
            xaxis_title="x (m)", yaxis_title="M (N·m)",
            height=440, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_shear:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ser["Vz_N"], mode="lines", name="Vz (plano XY → Fz)"))
        fig.add_trace(go.Scatter(x=x, y=ser["Vy_N"], mode="lines", name="Vy (plano XZ → Fy)"))
        fig.update_layout(
            xaxis_title="x (m)", yaxis_title="V (N)",
            height=440, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_reac:
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Plano XY → forças em Z (solver com Iy)**")
            if reac["reac_z"]:
                st.dataframe(reactions_table(reac["reac_z"]), use_container_width=True, hide_index=True)
            else:
                st.info("Sem carregamentos no plano XY (Z).")

        with cR:
            st.markdown("**Plano XZ → forças em Y (solver com Iz)**")
            if reac["reac_y"]:
                st.dataframe(reactions_table(reac["reac_y"]), use_container_width=True, hide_index=True)
            else:
                st.info("Sem carregamentos no plano XZ (Y).")

    with tab_check:
        sigma_vm = pk["sigma_vm_max_Pa"] / 1e6
        sigma_adm = pk["sigma_adm_Pa"] / 1e6
        fy = meta["fy"] / 1e6

        c1, c2, c3 = st.columns(3)
        c1.metric("σ_vm,max", f"{sigma_vm:.2f} MPa")
        c2.metric("σ_adm = fy/FS", f"{sigma_adm:.2f} MPa")
        c3.metric("fy", f"{fy:.2f} MPa")

        st.write(f"σvm ≤ σadm: {'OK ✅' if pk['ok_sigma_adm'] else 'NÃO OK ❌'}")
        st.write(f"σvm ≤ fy: {'OK ✅' if pk['ok_yield'] else 'NÃO OK ❌'}")

# ============================================================
# PÁGINA: MEMORIAL
# ============================================================
def render_memorial_page():
    st.subheader("Memorial de Cálculo (síntese técnica)")
    st.markdown(
        """
**Modelo adotado**
- Viga de Euler–Bernoulli (seções planas permanecem planas).
- Pequenas deformações.
- Flexão em dois planos independentes (XY → força em Z; XZ → força em Y).

**Formulação (MEF 1D)**
- Elemento com DOFs por nó: deslocamento transversal *v* e rotação *θ*.
- Matriz de rigidez do elemento: \\( \\mathbf{k_e} = \\frac{EI}{L^3} \\cdot \\mathbf{K} \\).

**Tensões**
- Flexão biaxial aproximada por:
\\[
\\sigma(y,z) = \\frac{M_y\\,z}{I_y} + \\frac{M_z\\,y}{I_z}
\\]
- Avaliação em pontos extremos (cantos) como aproximação conservadora.

**Critérios**
- Deslocamento admissível: \\( \\delta_{adm} = L / n \\), com \\(n\\in\\{200,250,300,400\\}\\).
- Tensão admissível: \\( \\sigma_{adm} = f_y/FS \\).
- Checagem de escoamento: \\( \\sigma_{vm} \\le f_y \\) (flexão pura → \\( \\tau \\approx 0\\)).

**Limitações**
- Não calcula torção (cargas excêntricas geram torção não contabilizada).
- Não considera Timoshenko (cisalhamento).
- Não considera flambagem / instabilidade global.
        """
    )

# ============================================================
# PÁGINA: SOBRE
# ============================================================
def render_about_page():
    st.subheader("Sobre")
    st.markdown(
        """
**Aplicativo:** Análise de flexão em vigas (biaxial)  
**Tecnologia:** Python + Streamlit + Plotly  
**Objetivo:** ferramenta técnica para cálculo rápido e visualização de resultados.

Sugestões de evolução:
- Inclusão de torção (G·J) e rotação θx
- Biblioteca de perfis (CSV/Excel) com validações
- Exportação de relatório (PDF/HTML)
- Suporte a Timoshenko em vigas curtas/altas (cisalhamento)
        """
    )

# ============================================================
# ROUTER
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
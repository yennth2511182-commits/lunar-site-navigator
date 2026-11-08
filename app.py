import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import date
from skyfield.api import PlanetaryConstants, load

st.set_page_config(page_title="Bản đồ Mặt Trăng 3D", layout="wide")
SOLAR_CONSTANT = 1361.0  # W/m2, mô hình đơn giản

# Tọa độ cần đối chiếu nguồn chính thức (NASA/LROC) trước khi nộp bài
SITES = {
    'Apollo 11': (0.674, 23.473),
    'Apollo 15': (26.132, 3.634),
    'Apollo 17': (20.191, 30.772),
    "Chang'e-4": (-45.457, 177.6),
    'Aristarchus': (26.3, -46.8),
}


@st.cache_resource
def setup():
    ts = load.timescale()
    eph = load('de421.bsp')
    pc = PlanetaryConstants()
    pc.read_text(load('moon_080317.tf'))
    pc.read_text(load('pck00008.tpc'))
    pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))
    frame = pc.build_frame_named('MOON_ME_DE421')
    return ts, eph, pc, frame


ts, eph, pc, frame = setup()
earth, moon, sun = eph['earth'], eph['moon'], eph['sun']


def unit_normal(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def moon_frame_dir(body, t, transpose):
    """Hướng (vector đơn vị) tới thiên thể, trong hệ tọa độ gắn với Mặt Trăng."""
    pos = moon.at(t).observe(body).position.km
    R = frame.rotation_at(t)
    if transpose:
        R = np.swapaxes(R, 0, 1)
    v = np.einsum('ij...,j...->i...', R, pos)
    return v / np.linalg.norm(v, axis=0)


def skyfield_sun_alt(lat, lon, t):
    site = moon + pc.build_latlon_degrees(frame, lat, lon)
    alt, _, _ = site.at(t).observe(sun).apparent().altaz()
    return alt.degrees


# Tự kiểm tra quy ước hệ tọa độ bằng hàm đã đối chiếu với Horizons
t0 = ts.utc(2026, 11, 15, 12, 0)
ref = skyfield_sun_alt(10.0, 20.0, t0)
diffs = {}
for tr in (False, True):
    d_ = moon_frame_dir(sun, t0, tr)
    est = np.degrees(np.arcsin(np.clip(unit_normal(10.0, 20.0) @ d_, -1, 1)))
    diffs[tr] = abs(est - ref)
TRANSPOSE = min(diffs, key=diffs.get)
if diffs[TRANSPOSE] > 1.0:
    st.warning(f"Kiểm tra nội bộ lệch {diffs[TRANSPOSE]:.1f}°: kết quả trang này chưa đáng tin.")

# ---------- Thanh bên ----------
d = st.sidebar.date_input("Ngày (UTC)", date(2026, 11, 15),
                          min_value=date(1950, 1, 1), max_value=date(2049, 12, 31))
hour = st.sidebar.slider("Giờ (UTC)", 0, 23, 12)
days = st.sidebar.slider("Số ngày để tính % có nắng", 5, 60, 30)
t_now = ts.utc(d.year, d.month, d.day, hour)

if 'sel' not in st.session_state:
    st.session_state.sel = SITES['Apollo 11']

# ---------- Bản đồ chiếu sáng ----------
lats = np.arange(-90, 91, 5)
lons = np.arange(-180, 181, 5)
LON, LAT = np.meshgrid(lons, lats)
N = unit_normal(LAT, LON)
sun_dir = moon_frame_dir(sun, t_now, TRANSPOSE)
earth_dir = moon_frame_dir(earth, t_now, TRANSPOSE)
flux = SOLAR_CONSTANT * np.clip(N @ sun_dir, 0, None)

st.title("Bản đồ chiếu sáng Mặt Trăng")
st.caption("Nhấp vào một điểm trên bản đồ (hoặc một điểm hạ cánh) để chọn. "
           "Năng lượng tính cho mặt phẳng nằm ngang, chưa có địa hình và hiệu suất pin.")

left, right = st.columns([3, 2])

with left:
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(
        x=LON.ravel(), y=LAT.ravel(), mode='markers', name='Lưới',
        marker=dict(size=8, color=flux.ravel(), colorscale='YlOrRd', cmin=0,
                    cmax=SOLAR_CONSTANT, colorbar=dict(title='W/m²')),
        hovertemplate='Vĩ độ %{y}°, Kinh độ %{x}°<extra></extra>'))
    fig2d.add_trace(go.Scatter(
        x=[v[1] for v in SITES.values()], y=[v[0] for v in SITES.values()],
        mode='markers+text', text=list(SITES), textposition='top center',
        marker=dict(size=11, color='royalblue', symbol='diamond'), name='Điểm hạ cánh'))
    fig2d.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title='Kinh độ Đông (độ)', yaxis_title='Vĩ độ (độ)',
                        showlegend=False)
    fig2d.update_xaxes(range=[-185, 185])
    fig2d.update_yaxes(range=[-95, 95])
    event = st.plotly_chart(fig2d, on_select='rerun', selection_mode='points',
                            key='moonmap', use_container_width=True)
    pts = event.selection.points if event and event.selection else []
    if pts:
        st.session_state.sel = (float(pts[0]['y']), float(pts[0]['x']))

if st.sidebar.checkbox("Nhập tọa độ bằng tay"):
    la0, lo0 = st.session_state.sel
    la_in = st.sidebar.number_input("Vĩ độ", -90.0, 90.0, float(la0))
    lo_in = st.sidebar.number_input("Kinh độ Đông", -180.0, 180.0, float(lo0))
    st.session_state.sel = (la_in, lo_in)

lat, lon = st.session_state.sel

# ---------- Chỉ số tại điểm đã chọn ----------
n_sel = unit_normal(lat, lon)
sin_alt_now = float(n_sel @ sun_dir)
sun_alt_now = float(np.degrees(np.arcsin(np.clip(sin_alt_now, -1, 1))))
incidence = 90.0 - sun_alt_now
flux_now = SOLAR_CONSTANT * max(sin_alt_now, 0.0)
earth_alt_now = float(np.degrees(np.arcsin(np.clip(n_sel @ earth_dir, -1, 1))))

minutes = np.arange(0, days * 24 * 60, 60)
t_series = ts.utc(d.year, d.month, d.day, 0, minutes)
sdir = moon_frame_dir(sun, t_series, TRANSPOSE)
edir = moon_frame_dir(earth, t_series, TRANSPOSE)
sin_alt_s = n_sel @ sdir
pct_sun = float((sin_alt_s > 0).mean() * 100)
mean_flux = float((SOLAR_CONSTANT * np.clip(sin_alt_s, 0, None)).mean())
pct_earth = float(((n_sel @ edir) > 0).mean() * 100)

with right:
    st.subheader("Điểm đã chọn")
    c1, c2 = st.columns(2)
    c1.metric("Vĩ độ", f"{lat:.2f}°")
    c2.metric("Kinh độ Đông", f"{lon:.2f}°")
    c1.metric("Góc cao Mặt Trời", f"{sun_alt_now:.1f}°")
    c2.metric("Góc tới (so với phương thẳng đứng)", f"{incidence:.1f}°" if sun_alt_now > 0 else "—")
    c1.metric("Năng lượng hiện tại", f"{flux_now:.0f} W/m²")
    c2.metric("Năng lượng TB / chu kỳ", f"{mean_flux:.0f} W/m²")
    c1.metric(f"% thời gian có nắng ({days} ngày)", f"{pct_sun:.0f}%")
    c2.metric("% thời gian thấy Trái Đất", f"{pct_earth:.0f}%")
    st.write(f"**Trái Đất lúc này:** {'trên' if earth_alt_now > 0 else 'dưới'} chân trời "
             f"({earth_alt_now:.1f}°)")
    chk = abs(sun_alt_now - skyfield_sun_alt(lat, lon, t_now))
    st.caption(f"Kiểm tra chéo với hàm góc cao đã đối chiếu Horizons: lệch {chk:.2f}°.")

    # ---------- Quả cầu 3D ----------
    surf = go.Surface(x=N[..., 0], y=N[..., 1], z=N[..., 2], surfacecolor=flux,
                      colorscale='YlOrRd', cmin=0, cmax=SOLAR_CONSTANT, showscale=False,
                      hoverinfo='skip')
    fig3d = go.Figure(surf)
    sx, sy, sz = (np.array([[n[i]] for n in [n_sel]]).ravel() * 1.03)
    fig3d.add_trace(go.Scatter3d(x=[sx], y=[sy], z=[sz], mode='markers',
                                 marker=dict(size=6, color='black'), name='Điểm chọn'))
    for nm, v, col in [('Mặt Trời', sun_dir, 'orange'), ('Trái Đất', earth_dir, 'royalblue')]:
        fig3d.add_trace(go.Scatter3d(x=[0, 1.8 * v[0]], y=[0, 1.8 * v[1]], z=[0, 1.8 * v[2]],
                                     mode='lines+markers', name=nm,
                                     line=dict(color=col, width=5), marker=dict(size=4)))
    for nm, (la_s, lo_s) in SITES.items():
        p = unit_normal(la_s, lo_s) * 1.02
        fig3d.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode='markers',
                                     marker=dict(size=4, color='royalblue'), name=nm,
                                     showlegend=False, hovertext=nm, hoverinfo='text'))
    fig3d.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(aspectmode='data', xaxis_visible=False,
                                   yaxis_visible=False, zaxis_visible=False))
    st.plotly_chart(fig3d, use_container_width=True)

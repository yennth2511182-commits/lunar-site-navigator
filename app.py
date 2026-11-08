import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date
from skyfield.api import PlanetaryConstants, load

st.set_page_config(page_title="Lunar Site Navigator", layout="wide")


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


def sky_position(lat, lon, t):
    site = moon + pc.build_latlon_degrees(frame, lat, lon)
    out = {}
    for name, body in [('sun', sun), ('earth', earth)]:
        alt, az, _ = site.at(t).observe(body).apparent().altaz()
        out[name] = (alt.degrees, az.degrees)
    return out


def next_event(alt, minutes):
    up = alt > 0
    idx = np.where(up != up[0])[0]
    if len(idx) == 0:
        return ('luôn trên chân trời' if up[0] else 'luôn dưới chân trời'), None
    return ('đang trên, sắp lặn' if up[0] else 'đang dưới, sắp mọc'), round(minutes[idx[0]] / 60, 1)


def sky_plot(sun_pos, earth_pos):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(['90°', '60°', '30°', '0° (chân trời)'])
    for name, (alt, az), color in [('Mặt Trời', sun_pos, 'orange'),
                                   ('Trái Đất', earth_pos, 'tab:blue')]:
        if alt > 0:
            ax.scatter(np.radians(az), 90 - alt, s=250, color=color, label=name, zorder=3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='lower left', bbox_to_anchor=(0.85, 0))
    return fig


# Tọa độ: cần đối chiếu với nguồn chính thức (NASA/LROC) và ghi nguồn trong bài
SITES = {
    'Apollo 11': (0.674, 23.473),
    'Apollo 15': (26.132, 3.634),
    'Apollo 17': (20.191, 30.772),
    "Chang'e-4 (mặt xa)": (-45.457, 177.6),
}

st.title("Lunar Landing Site Navigator")
st.caption("Góc cao Mặt Trời / Trái Đất tại điểm hạ cánh. Đường chân trời coi là phẳng (chưa có địa hình).")

d = st.sidebar.date_input("Ngày bắt đầu (UTC)", date(2026, 11, 15),
                          min_value=date(1950, 1, 1), max_value=date(2049, 12, 31))
days = st.sidebar.slider("Số ngày xem", 5, 60, 30)
hour = st.sidebar.slider("Thời điểm xem (giờ kể từ ngày bắt đầu)", 0, days * 24, 0)
t_now = ts.utc(d.year, d.month, d.day, hour)


def site_picker(label, default):
    choice = st.sidebar.selectbox(label, list(SITES) + ['Tự nhập tọa độ'], index=default)
    if choice == 'Tự nhập tọa độ':
        lat = st.sidebar.number_input(f"{label}: vĩ độ", -90.0, 90.0, 0.0, key=label + 'lat')
        lon = st.sidebar.number_input(f"{label}: kinh độ Đông", -180.0, 180.0, 0.0, key=label + 'lon')
        return 'Tùy chọn', lat, lon
    return (choice, *SITES[choice])


minutes = np.arange(0, days * 24 * 60, 30)
t = ts.utc(d.year, d.month, d.day, 0, minutes)
x = minutes / 60 / 24

cols = st.columns(2)
for col, (label, default) in zip(cols, [("Điểm A", 0), ("Điểm B", 3)]):
    name, lat, lon = site_picker(label, default)
    pos = sky_position(lat, lon, t)
    sun_alt, earth_alt = pos['sun'][0], pos['earth'][0]
    with col:
        st.subheader(f"{label}: {name}")
        c1, c2 = st.columns(2)
        c1.metric("Mặt Trời trên chân trời", f"{(sun_alt > 0).mean() * 100:.0f}%")
        c2.metric("Trái Đất trên chân trời", f"{(earth_alt > 0).mean() * 100:.0f}%")
        for body, alt in [('Mặt Trời', sun_alt), ('Trái Đất', earth_alt)]:
            state, h = next_event(alt, minutes)
            st.write(f"**{body}:** {state}" + (f", đổi trạng thái sau {h} giờ" if h else ""))

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, sun_alt, label='Mặt Trời')
        ax.plot(x, earth_alt, label='Trái Đất')
        ax.axvline(hour / 24, color='gray', ls='--', lw=1)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Góc cao (độ)')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        now = sky_position(lat, lon, t_now)
        st.write("**Bầu trời tại thời điểm đã chọn**")
        sky_fig = sky_plot(now['sun'], now['earth'])
        st.pyplot(sky_fig)
        plt.close(sky_fig)

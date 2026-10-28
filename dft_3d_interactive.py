#!/usr/bin/env python3
"""Interactive DFT geometry using NumPy + Matplotlib (MATLAB-style plotting).
Install: python -m pip install numpy matplotlib
Run:     python dft_3d_interactive.py
Notebook: use %matplotlib widget first (requires ipympl).
Drag the 3D axes to rotate; use the toolbar to zoom or save.
--snapshot PATH renders a preview without opening a GUI.
--self-test checks orthogonality, reconstruction and slider callbacks.
"""
import argparse
import numpy as np
import matplotlib


def coefficient(nu, f0=3.0, N=16):
    n = np.arange(N)
    return np.mean(np.exp(2j*np.pi*(f0-nu)*n/N))


class DFTExplorer:
    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
        self.plt = plt
        self.N = 16
        self.running = False
        self.fig = plt.figure(figsize=(14, 9), facecolor='#fafaf7')
        self.fig.suptitle('DISCRETE FOURIER TRANSFORM | rotating phasors', fontsize=18, weight='bold')
        self.fig.text(.5, .925, r'$x[n]=e^{i2\pi f_0 n/16}$     $C(\nu)=\frac{1}{16}\sum_{n=0}^{15}x[n]e^{-i2\pi\nu n/16}$', ha='center', fontsize=17)
        self.ax3 = self.fig.add_axes([.035, .29, .57, .59], projection='3d')
        self.axc = self.fig.add_axes([.68, .55, .25, .30])
        self.axs = self.fig.add_axes([.67, .30, .28, .17])
        self.snu = Slider(self.fig.add_axes([.16, .19, .68, .025]), 'Trial frequency ν', 0, 15, valinit=2.83, valfmt='%.2f')
        self.sf0 = Slider(self.fig.add_axes([.16, .135, .68, .025]), 'Input cycles f₀', 0, 15, valinit=3, valfmt='%.2f')
        self.selev = Slider(self.fig.add_axes([.16, .08, .29, .02]), 'Elevation', -80, 80, valinit=20)
        self.saz = Slider(self.fig.add_axes([.59, .08, .25, .02]), 'Azimuth', -180, 180, valinit=-65)
        self.buttons = []
        for i, (label, callback) in enumerate([
            ('ν = 0', lambda e: self.snu.set_val(0)),
            ('ν = 2', lambda e: self.snu.set_val(2)),
            ('Match f₀', lambda e: self.snu.set_val(self.sf0.val)),
            ('Play / Pause', self.toggle),
            ('Reset', self.reset),
        ]):
            b = Button(self.fig.add_axes([.16+i*.143, .025, .132, .032]), label)
            b.on_clicked(callback)
            self.buttons.append(b)
        self.snu.on_changed(self.update)
        self.sf0.on_changed(self.update)
        self.selev.on_changed(self.camera)
        self.saz.on_changed(self.camera)
        self.ax3.view_init(elev=20, azim=-65)
        self.timer = self.fig.canvas.new_timer(interval=80)
        self.timer.add_callback(self.tick)
        self.update()

    def camera(self, _=None):
        self.ax3.view_init(elev=self.selev.val, azim=self.saz.val)
        self.fig.canvas.draw_idle()

    def toggle(self, _=None):
        self.running = not self.running
        self.timer.start() if self.running else self.timer.stop()

    def tick(self):
        self.snu.set_val((self.snu.val + .04) % 15)

    def reset(self, _=None):
        self.running = False
        self.timer.stop()
        for s in (self.snu, self.sf0, self.selev, self.saz):
            s.reset()
        self.camera()

    def update(self, _=None):
        n = np.arange(self.N)
        t = np.linspace(0, 15, 800)
        f0, nu = self.sf0.val, self.snu.val
        z = np.exp(2j*np.pi*(f0-nu)*n/16)
        curve = np.exp(2j*np.pi*(f0-nu)*t/16)
        original = np.exp(2j*np.pi*f0*t/16)
        C = z.mean()
        ax = self.ax3
        elev, azim = ax.elev, ax.azim
        ax.clear()
        ax.plot(t, original.real, original.imag, color='#7381bc', alpha=.32, label='Original input')
        ax.plot(t, curve.real, curve.imag, color='#367d6e', lw=2.2, label='After mixing')
        ax.scatter(n, z.real, z.imag, color='#367d6e', s=23)
        plane = 17
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.full_like(theta, plane), np.cos(theta), np.sin(theta), color='#abb9b1', lw=1)
        for value in z:
            ax.plot([plane, plane], [0, value.real], [0, value.imag], color='#367d6e', alpha=.16)
        ax.scatter(np.full(16, plane), z.real, z.imag, s=22, color='#367d6e')
        ax.quiver(plane, 0, 0, 0, C.real, C.imag, color='#b54236', linewidth=3, arrow_length_ratio=.13)
        ax.scatter([plane], [C.real], [C.imag], color='#b54236', s=48)
        ax.plot([0, plane], [0, 0], [0, 0], color='gray', alpha=.4)
        ax.set(xlim=(0, 18), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2), xlabel='Sample n', ylabel='Re', zlabel='Im')
        ax.set_box_aspect((2.5, 1.3, 1.3))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'ν = {nu:.2f}   |   residual cycles = {f0-nu:.2f}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(False)
        a = self.axc
        a.clear()
        a.plot(np.cos(theta), np.sin(theta), color='#bdc9c2')
        a.axhline(0, color='gray', lw=.6)
        a.axvline(0, color='gray', lw=.6)
        for v in z:
            a.plot([0, v.real], [0, v.imag], color='#367d6e', alpha=.18)
        a.scatter(z.real, z.imag, color='#367d6e', s=24)
        a.annotate('', xy=(C.real, C.imag), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#b54236', lw=2.5))
        a.scatter([C.real], [C.imag], color='#b54236', s=35, zorder=4)
        a.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xlabel='Re', ylabel='Im', title=f'Phasor average: |C| = {abs(C):.4f}')
        a.set_aspect('equal')
        a.text(.02, 1.12, f'C = {C.real:+.4f} {C.imag:+.4f}i', transform=a.transAxes, color='#b54236')
        freq = np.linspace(0, 15, 601)
        response = np.abs(np.exp(2j*np.pi*(f0-freq[:, None])*n/16).mean(axis=1))
        bins = np.arange(16)
        spectrum = np.abs(np.fft.fft(np.exp(2j*np.pi*f0*n/16))/16)
        a = self.axs
        a.clear()
        a.plot(freq, response, color='#367d6e', lw=1.5, label='Continuous trial ν')
        a.stem(bins, spectrum, linefmt='C0-', markerfmt='C0o', basefmt=' ')
        a.scatter([nu], [abs(C)], color='#b54236', zorder=5)
        a.set(xlim=(0, 15), ylim=(-.04, 1.08), xlabel='Frequency bin / cycles per record', ylabel='|C|')
        a.set_title('Curve: trial frequencies • dots: DFT bins', fontsize=10)
        a.grid(alpha=.15)
        self.fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', metavar='PNG')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.snapshot or args.self_test:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model = DFTExplorer()
    if args.self_test:
        for k in range(16):
            assert np.isclose(coefficient(k), 1 if k == 3 else 0, atol=1e-13)
        n = np.arange(16)
        x = np.exp(2j*np.pi*3.24*n/16)
        assert np.allclose(np.fft.ifft(np.fft.fft(x)), x)
        for v in (0, 2, 2.83, 3, 15):
            model.snu.set_val(v)
        model.sf0.set_val(3.4)
        model.reset()
        print('PASS: orthogonality, FFT reconstruction, slider callbacks and reset')
    if args.snapshot:
        model.fig.savefig(args.snapshot, dpi=150)
        print(args.snapshot)
    if not (args.snapshot or args.self_test):
        plt.show()
    else:
        plt.close(model.fig)


if __name__ == '__main__':
    main()

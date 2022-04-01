import tkinter as tk
import numpy as np
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from SocialNetwork import SocialNetwork

top_choices = ['Complete', 'Small World', 'Scale Free', 'Random']
tf_choices = ['True', 'False']
speed_choices = ['Slow', 'Normal', 'Fast']

class SNGUI:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title('Social Network Evolution Engine')

        self.graph = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.points = None
        self.pos = None
        self.x, self.y = list(), list()
        self.texts, self.lines = list(), dict()
        self.drawing = False
        self.numsteps = 0
        self.clearflag = False

        self.vars = {'n_in': tk.IntVar(),
                     'fr_in': tk.DoubleVar(),
                     'unf_in': tk.DoubleVar(),
                     'upd_in': tk.DoubleVar(),
                     'sat_in': tk.IntVar(),
                     'topology_in': tk.StringVar(),
                     'extreme_in': tk.StringVar(),
                     'sym_in': tk.StringVar(),
                     'labels_in': tk.StringVar(),
                     'nodepos_in': tk.StringVar(),
                     'speed_in': tk.StringVar(),
                     'readfile_in': tk.StringVar(),
                     'writefile_in': tk.StringVar(),
                     'ops_in': tk.StringVar()}

        self.boxes = {}
        self.vals = {}

        self.vars['topology_in'].set('Small World')
        self.vars['n_in'].set(3)
        self.vars['fr_in'].set(0.)
        self.vars['unf_in'].set(0.)
        self.vars['extreme_in'].set('True')
        self.vars['sym_in'].set('True')
        self.vars['labels_in'].set('True')
        self.vars['nodepos_in'].set('True')
        self.vars['upd_in'].set(1.)
        self.vars['sat_in'].set(2)
        self.vars['speed_in'].set('Fast')

        self.frame = tk.Frame(self.root, padx=5, pady=5)
        self.frame.grid(padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)

        self.frame1 = tk.LabelFrame(self.frame, text='Network Properties')
        self.frame1.grid(row=1, padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)

        self.frame2 = tk.LabelFrame(self.frame, text='Plot Properties')
        self.frame2.grid(row=2, padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)

        self.frame3 = tk.LabelFrame(self.frame, text='Actions')
        self.frame3.grid(row=3, padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)

        # self.frame4 = tk.LabelFrame(self.frame, text='Save/Load')
        # self.frame4.grid(row=4, padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)

        self.frame5 = tk.LabelFrame(self.frame, text='Status')
        self.frame5.grid(row=5, padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)

        self.init_entrybox(self.frame1, text='Number of nodes: ', row=0, prefix='n')
        self.init_entrybox(self.frame1, text='Probability to create new edge: ', row=1, prefix='fr')
        self.init_entrybox(self.frame1, text='Probability of destroying an existing edge: ', row=2, prefix='unf')
        self.init_entrybox(self.frame1, text='Probability of adopting neighboring opinion: ', row=3, prefix='upd')
        self.init_entrybox(self.frame1, text='Average node degree: ', row=4, prefix='sat')
        self.init_dropdown(self.frame1, text='Topology', row=5, prefix='topology', choices=top_choices)
        self.init_dropdown(self.frame1, text='Initialize opinions at extremes', row=6, prefix='extreme', choices=tf_choices)
        self.init_dropdown(self.frame1, text='Symmetric edges', row=7, prefix='sym', choices=tf_choices)
        # self.init_dropdown(self.frame2, text='Node labels on plot', row=0, prefix='labels', choices=tf_choices)
        self.init_dropdown(self.frame2, text='Static node positions', row=1, prefix='nodepos', choices=tf_choices)
        self.init_dropdown(self.frame2, text='Animation speed', row=2, prefix='speed', choices=speed_choices)

        tk.Button(self.frame3, command=self.construct_network, text='Construct Network', width=18).grid(row=0, padx=5, pady=5,
                                                                                                        sticky=tk.E)
        tk.Button(self.frame3, command=self.clear, text='Clear', width=18).grid(row=0, column=1, padx=5, pady=5,
                                                                               sticky=tk.W)
        tk.Button(self.frame3, command=self.advance, text='Advance', width=18).grid(row=1, padx=5, pady=5,
                                                                                   sticky=tk.E)
        # tk.Button(self.frame3, command=self.advance10, text='Advance 10', width=18).grid(row=1, column=1, padx=5, pady=5,
        #                                                                                sticky=tk.W)
        tk.Button(self.frame3, command=self.init_animation, text='Animate', width=18).grid(row=1, column=1, padx=5, pady=5,
                                                                                           sticky=tk.W)
        tk.Button(self.frame3, command=self.stop_animation, text='Stop', width=18).grid(row=2, padx=5, pady=5,
                                                                                        sticky=tk.E)
        # tk.Button(self.frame3, command=self.write, text='Write matrix', width=18).grid(row=2, column=1, padx=5, pady=5,
        #                                                                                sticky=tk.W)

        # self.init_entrybox(self.frame4, text='Input file: ', row=0, prefix='readfile', boxwidth=45)
        # self.init_entrybox(self.frame4, text='Output file: ', row=1, prefix='writefile', boxwidth=45)
        # self.init_entrybox(self.frame4, text='Initial opinions file: ', row=2, prefix='ops', boxwidth=45)

        self.status = tk.Label(self.frame5, text='Ready for a command!', fg='green')
        self.status.grid(padx=10, pady=10)

        self.plotframe = tk.Frame(self.root, padx=5, pady=5)
        self.plotframe.grid(padx=5, pady=5, row=0, column=1)

        self.init_plot()

        self.root.mainloop()

    def set_status(self, message, color='green'):
        """

        :param message:
        :param color:
        :return:
        """
        self.status.config(text=message, fg=color)

    def init_entrybox(self, container, text='', row=0, prefix='', boxwidth=10):
        tk.Label(container, text=text).grid(row=row, column=0, padx=5, pady=5, sticky=tk.E)
        self.boxes[f'{prefix}_entry'] = tk.Entry(container, width=boxwidth,
                                                 textvariable=self.vars[f'{prefix}_in'])
        self.boxes[f'{prefix}_entry'].grid(row=row, column=1, padx=5, pady=5)

    def init_dropdown(self, container, text='', row=0, prefix='', choices=[]):
        tk.Label(container, text=text).grid(row=row, column=0, padx=5, pady=5, sticky=tk.E)
        self.boxes[f'{prefix}_entry'] = tk.OptionMenu(container, self.vars[f'{prefix}_in'], *choices)
        self.boxes[f'{prefix}_entry'].grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)

    def init_plot(self):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plotframe)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, self.plotframe)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def get_vals(self):
        for key in self.vars:
            val = self.vars[key].get()
            if val in tf_choices:
                self.vals[key] = True if val == 'True' else False
            else:
                self.vals[key] = val

    def check_input(self):
        flag = True
        if self.vals['n_in'] < 1 or self.vals['n_in'] < self.vals['sat_in']:
            self.set_status('Please increase the number of nodes.', 'red')
            flag = False
        elif not 0. <= self.vals['unf_in'] <= 1.:
            self.set_status('Enter an edge destruction probability between 0 and 1.', 'red')
            flag = False
        elif not 0. <= self.vals['fr_in'] <= 1.:
            self.set_status('Enter an edge creation probability between 0 and 1.', 'red')
            flag = False
        elif not 0. <= self.vals['upd_in'] <= 1.:
            self.set_status('Enter a state update probability between 0 and 1.', 'red')
            flag = False
        return flag

    def get_labels(self):
        """

        :return:
        """
        return [str(i) + ':' + str(round(self.graph.prop('attribute_space')[i][0], 3))
                for i in range(self.graph.prop('n'))]

    def get_colors(self):
        """

        :return:
        """
        colors = []
        nums = [i[0] for i in self.graph.prop('attribute_space')]
        for i in range(self.graph.prop('n')):
            # dist = (nums[i] + 1) / 2
            dist = nums[i]
            rgb = (dist, 0., 1 - dist, .25)
            colors.append(rgb)
        return colors

    def plot_network(self, added=[], removed=[]):
        """

        :return:
        """
        labels = self.get_labels()
        colors = self.get_colors()

        # If there are no positions defined or positions need to change over time, calculate new ones
        if self.pos is None or not self.vals['nodepos_in']:
            self.x, self.y, pos = self.graph.get_xy(initial_pos=self.pos)
            self.pos = pos

        # If there are no dots on the screen
        if self.points is None:

            # Draw points
            self.points = self.ax.scatter(self.x, self.y, s=350, alpha=.5, c=colors)

            # Draw edges
            for e in self.graph._graph.edges():
                if (e[0], e[1]) in self.lines or (e[1], e[0]) in self.lines:
                    continue
                line, = self.ax.plot([self.x[e[0]], self.x[e[1]]],
                                     [self.y[e[0]], self.y[e[1]]], 'k', alpha=.25)
                self.lines[e] = line

            # Draw labels if wanted
            if self.vals['labels_in']:
                for i in range(len(labels)):
                    self.texts.append(self.ax.text(self.x[i], self.y[i], labels[i]))
            else:
                for i in range(len(labels)):
                    self.texts.append(self.ax.text(self.x[i], self.y[i], ''))

        # If there are dots on the screen
        else:

            # Remove any deleted edges if the graph is dynamic
            for (u, v) in removed:
                if (u, v) in self.lines:
                    self.ax.lines.remove(self.lines[(u, v)])
                    del self.lines[(u, v)]
                if self.graph.prop('symmetric'):
                    if (v, u) in self.lines:
                        try:
                            self.ax.lines.remove(self.lines[(v, u)])
                        except:
                            pass
                        del self.lines[(v, u)]

            # Add any new edges if the graph is dynamic
            for (u, v) in added:

                # Skip if the line is already drawn
                if (u, v) in self.lines or (v, u) in self.lines:
                    continue
                # Otherwise, draw the edge
                line, = self.ax.plot([self.x[u], self.x[v]],
                                     [self.y[u], self.y[v]], 'k', alpha=.25)
                self.lines[(u, v)] = line

                # If the graph is symmetric, make two edge entries point to the same line
                if self.graph.prop('symmetric'):
                    self.lines[(v, u)] = self.lines[(u, v)]

            # Clean up by drawing any remaining edges
            for e in self.graph._graph.edges():
                if e not in self.lines and (e[1], e[0]) not in self.lines:
                    line, = self.ax.plot([self.x[e[0]], self.x[e[1]]],
                                         [self.y[e[0]], self.y[e[1]]], 'k', alpha=.25)
                    self.lines[e] = line
                else:
                    try:
                        self.lines[e].set_xdata([self.x[e[0]], self.x[e[1]]])
                        self.lines[e].set_ydata([self.y[e[0]], self.y[e[1]]])
                    except KeyError:
                        newe = (e[1], e[0])
                        self.lines[newe].set_xdata([self.x[newe[0]], self.x[newe[1]]])
                        self.lines[newe].set_ydata([self.y[newe[0]], self.y[newe[1]]])
            self.points.set_offsets(np.c_[self.x, self.y])
            self.points.set_color(colors)
            if self.vals['labels_in']:
                for i in range(len(labels)):
                    self.texts[i].set_text(labels[i])
                    self.texts[i].set_position((self.x[i], self.y[i]))
            else:
                for i in range(len(labels)):
                    self.texts[i].set_text('')

        curr_lines = self.lines.values()
        for line in self.ax.lines:
            if line not in curr_lines:
                self.ax.lines.remove(line)

        # update ax.viewLim using the new dataLim
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])

        self.canvas.draw()

        self.set_status(f'Step: {self.numsteps}', 'black')

    def construct_network(self):
        """

        :return:
        """
        if self.graph is not None:
            self.set_status('Network already constructed!', 'red')
            return

        self.get_vals()
        props = {}

        props['n'] = self.vals['n_in']
        props['friend'] = self.vals['fr_in']
        props['unfriend'] = self.vals['unf_in']
        props['update'] = self.vals['upd_in']
        props['symmetric'] = self.vals['sym_in']
        props['topology'] = self.vals['topology_in'].lower()
        props['saturation'] = self.vals['sat_in'] / self.vals['n_in']
        props['directed'] = True
        props['attributes'] = 'continuous'
        if self.vals['readfile_in']:
            props['matrix_file'] = self.vals['readfile_in']

        if not self.check_input():
            return

        self.graph = SocialNetwork(props)
        # print(self.graph.prop('attribute_space'))

        if self.vals['ops_in']:
            myvals = []
            with open(self.vals['ops_in'], 'r') as f:
                myvals = [float(j) for j in f.readline()[:-1].split()]
            for j in range(len(myvals)):
                self.graph._properties['attribute_space'][j] = myvals[j]
        elif self.vals['extreme_in']:
            for i in range(self.graph.prop('n')):
                if self.graph.prop('attribute_space')[i][0] <= .5:
                    self.graph._properties['attribute_space'][i][0] = 0.
                    for j in range(self.graph.prop('n')):
                        self.graph._properties['masks'][j][i][0] = 0.
                else:
                    self.graph._properties['attribute_space'][i][0] = 1.
                    for j in range(self.graph.prop('n')):
                        self.graph._properties['masks'][j][i][0] = 1.

        self.set_status('Network created successfully!')
        self.plot_network()

    def write(self):
        self.get_vals()
        if self.graph is None:
            self.set_status('No graph to write!', 'red')
            return
        if self.vals['writefile_in'] == '':
            self.set_status('No filename provided!', 'red')
            return
        self.graph._write_adj_matrix(self.vals['writefile_in'])
        self.set_status(f'Graph written to {self.vals["writefile_in"]}.', 'green')

    def clear(self):
        """
        Clear the plot and all class attributes associated with it.
        :return: None
        """
        if self.drawing:
            self.drawing = False
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        self.graph = None
        self.x, self.y = None, None
        self.texts = list()
        self.points = None
        self.lines = dict()
        self.pos = None
        self.numsteps = 0
        self.clearflag = True
        self.set_status('Graph cleared.', 'green')

    def advance(self):
        """

        :return:
        """
        if self.graph is not None:
            self.numsteps += 1

            removed = self.graph.act()
            self.graph.update()
            added = self.graph.create_connections(triadic=False)

            self.get_vals()

            self.plot_network(added=added, removed=removed)
        else:
            self.set_status('No active network!', 'red')

    def advance10(self):
        """

        :return:
        """
        if self.graph is not None:
            for i in range(10):
                self.advance()
        else:
            self.set_status('No active network!', 'red')

    def init_animation(self):
        """

        :return:
        """
        self.drawing = True
        self.animate()
        self.root.update()

    def animate(self):
        """

        :return:
        """
        if self.graph is None and not self.clearflag:
            self.set_status('No graph to animate!', 'red')
            return

        self.clearflag = False
        speed = self.vals['speed_in']
        if self.drawing:
            self.advance()
            if speed == 'Slow':
                self.root.after(1000, self.animate)
            elif speed == 'Normal':
                self.root.after(500, self.animate)
            elif speed == 'Fast':
                self.root.after(100, self.animate)

    def stop_animation(self):
        """

        :return:
        """
        self.drawing = False

def main():
    SNGUI()


if __name__ == '__main__':
    main()

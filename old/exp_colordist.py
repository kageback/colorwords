import com_enviroments

def main():
    e = com_enviroments.make('wcs')

    e.print_color_map(f=lambda t: "{:.1f}".format(e.sim_index(1, t.index.values[0])[0]), pad=4)
    e.print_color_map(f=lambda t: "{:.1f}".format(e.sim_index(288, t.index.values[0])[0]), pad=4)
    e.print_color_map(f=lambda t: "{:.1f}".format(e.sim_index(289, t.index.values[0])[0]), pad=4)


if __name__ == "__main__":
    main()

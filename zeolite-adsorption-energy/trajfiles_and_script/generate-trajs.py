import xlrd
import ase.io
import pickle


class adsorption_case():
    """
    cases of DFT calculation
    """
    def __init__(self, zeolite, adsorbate, Etotal, casedir, slabdir,
                 zeolite_energy, adsorbate_energy):
        self.zeolite = zeolite
        self.adsorbate = adsorbate
        self.Etotal = Etotal
        self.casedir = casedir
        self.slabdir = slabdir
        self.zeolite_energy = zeolite_energy
        self.adsorbate_energy = adsorbate_energy

        self.adsorption_energy = self.Etotal - self.zeolite_energy - self.adsorbate_energy
        self.image = self.Get_traj_file()
        self.imagecheck()

    def imagecheck(self):
        if image != 0:
            adsorbate_check = []
            for atom in image:
                if atom.tag == 1:
                    adsorbate_check.append(atom.symbol)
            print('case:', self.casedir, 'Ead: ', self.adsorption_energy,
                  ': adsorbate atoms: ', self.adsorbate, 'adsorbates_real: ',
                  self.adsorbate, file=out_file)
        elif image == 0:
            print('case:', self.casedir, '*******can not find adsorbates****', file=out_file)


    def Get_traj_file(self):
        image_ad = ase.io.read(self.casedir)
        slabimage = ase.io.read(self.slabdir)
        image = Tag_adsorbate(image_ad, slabimage)
        if image != 0:
            image.wrap(pbc=[True, True, True])
            image.set_calculator(sp(atoms=image, energy=self.adsorption_energy))
        return image
        


def reorganazedir(casedir):
    casedir = casedir[19:]
    casedir = casedir.replace('\\', '-')
    casedir = casedir + '.traj'
    if casedir[0] == '-':
        casedir = casedir[1:]
    return casedir


def reorganazeadsorbate(adsorbate):
    adsorbate = adsorbate.replace('*', '')
    adsorbate = adsorbate.splite('+')
    return adsorbate

def Get_excel_data(excel):
    case_data = []
    zeolite_energy_dict = pickle.load(open('zeolite_energy.pkl', 'rb'))
    gas_energy_dict = pickle.load(open('gas_energy.pkl', 'rb'))
    zeolite_dir_dict = pickle.load(open('zeolite_dir.pkl', 'rb'))
    data = xlrd.open_workbook(excel)
    table = data.sheets()[0]
    rows = table.nrows
    for row in range(1, rows):
        zeolite = table.cell(row, 0).value
        adsorbate = table.cell(row, 3).value
        Etotal = table.cell(row, 5).value
        casedir = table.cell(row, 6).value
        slabdir = table.cell(row, 8).value
        if casedir == '' or adsorbate == '' or Etotal == '' or zeolite == '':
            continue
        casedir = reorganazedir(casedir)
        slabdir = zeolite_dir_dict[zeolite]
        adsorbate = reorganazeadsorbate(adsorbate)
        zeolite_energy = zeolite_energy_dict[zeolite]
        adsorbate_energy = 0
        for adsorbatei in adsorbate:
            adsorbate_energy += gas_energy_dict[adsorbatei]
        ad_case = adsorption_case(zeolite, adsorbate,
                                  Etotal, casedir, slabdir, zeolite_energy, adsorbate_energy)
        case_data.append[ad_case]
    return case_data

def Tag_adsorbate(image, slab_image):
    """
    mark the adsorbate atoms with tag = 1 (others are tag = 0)
    find the difference in atoms composition
    the adsorbates might be in the first or last
    to do: if this works not well, try to mark the atoms by comparing distances
    :param image:
    :param slab_image:
    :return:the image tagged or 0 (0 for non-ad / wrong images)
    """
    symbols_a = image.get_chemical_symbols()
    symbols_b = slab_image.get_chemical_symbols()

    ###################
    symbols_b_2 = slab_image.get_chemical_formula()
    symbols_a_1 = image[:-(len(symbols_a) - len(symbols_b))].get_chemical_formula()
    symbols_a_2 = image[(len(symbols_a) - len(symbols_b)):].get_chemical_formula()
    aaa = Get_tag_number(Mark_C_H(image))
    bbb = len(symbols_a) - len(symbols_b)
    ########
    if len(symbols_a) > len(symbols_b):
        if operator.eq(symbols_a[:-(len(symbols_a) - len(symbols_b))], symbols_b):
            for i in range(len(symbols_a) - 1, len(symbols_b) - 1, - 1):
                image[i].tag = 1
            return image
        elif operator.eq(symbols_a[(len(symbols_a) - len(symbols_b)):], symbols_b):
            for i in range(len(symbols_a)-len(symbols_b)):
                image[i].tag = 1
            return image
        # only CH
        elif Get_tag_number(Mark_C_H(image)) == len(symbols_a) - len(symbols_b):
            image = Mark_C_H(image)
            return image
        # H in slab
        elif symbols_b[-1] == symbols_a[len(symbols_b)] and operator.eq(symbols_a[:-(len(symbols_a) - len(symbols_b)+1)], symbols_b[:-1]):
            for i in range(len(symbols_a) - 1, len(symbols_b) - 2, - 1):
                image[i].tag = 1
            image[len(symbols_b)].tag = 0
            return image
        # change slab atom order

        elif operator.eq(symbols_b_2, symbols_a_1):
            for i in range(len(symbols_a) - 1, len(symbols_b) - 1, - 1):
                image[i].tag = 1
            return image
        elif operator.eq(symbols_b_2, symbols_a_2):
            for i in range(len(symbols_a)-len(symbols_b)):
                image[i].tag = 1
            return image


        else:
            return 0
    else:
        return 0


def Mark_C_H(image):
    """
    
    mark C and H as tag = 1, generate new, won't change image
    :param image:
    :return: image marked
    """
    image2 = image.copy()
    for atom in image2:
        if atom.symbol == 'C' or atom.symbol == 'H':
            atom.tag = 1
    return image2

def Get_tag_number(image):
    """
    get the number of atoms with tag = 1
    :param image:
    :return: number of atoms atom.tag = 1
    """
    number = 0
    for atom in image:
        if atom.tag == 1:
            number += 1
    return number


def main():
    global out_file
    out_file = open(('output_local.txt'), 'w')
    excel = 'lihuan-0505.xlsx'
    #databasedir = '/home/xiaoyan/zeolite/database/'
    databasedir = '/home/fuxiaoyan/zeolite/database'
    case_data = Get_excel_data(excel)
    images = []
    for case in case_data:
        if case.image != 0:
            images.append(case.image)
    ase.io.write('traj_taged_adsorptionenergy.traj', images)
    out_file.close()

if __name__=="__main__":
    main()

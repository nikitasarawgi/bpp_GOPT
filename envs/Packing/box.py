
class Box(object):
    def __init__(self, length, width, height, x, y, z):
        # dimension(x, y, z) + position(lx, ly, lz)
        self.size_x = length
        self.size_y = width
        self.size_z = height
        self.pos_x = x
        self.pos_y = y
        self.pos_z = z

    def standardize(self):
        """

        Returns:
            tuple(size + position)
        """
        return tuple([self.size_x, self.size_y, self.size_z, self.pos_x, self.pos_y, self.pos_z])



class DeformBox(object):
    def __init__(self, box_id, length, width, height, x, y, z, mass, spring_k, fragility = 0):
        # dimension(x, y, z) + position(lx, ly, lz)
        self.box_id = box_id
        self.size_x = length
        self.size_y = width
        self.size_z = height
        self.pos_x = x
        self.pos_y = y
        self.pos_z = z
        self.mass = mass
        self.spring_k = spring_k
        self.fragility = fragility

    def standardize(self):
        """

        Returns:
            tuple(box_id + size + position)
        """
        return tuple([self.box_id, self.size_x, self.size_y, self.size_z, self.pos_x, self.pos_y, self.pos_z])

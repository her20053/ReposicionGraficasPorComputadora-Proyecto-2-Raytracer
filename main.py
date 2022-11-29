from progress.bar import Bar
from math         import *

import sys
import struct
import random

class IntersectClass:
    def __init__(self, distance, point, normal, coords=None):
        self.distance = distance
        self.point = point
        self.normal = normal
        self.coords = coords

class ColorClass():
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __mul__(self, other):
        r = self.r
        g = self.g
        b = self.b

        if(type(other) == int or type(other) == float):
            r *= other
            g *= other
            b *= other
        else:
            r *= other.r
            g *= other.g
            b *= other.b

        r = min(255, max(r, 0))
        g = min(255, max(g, 0))
        b = min(255, max(b, 0))

        if r < 0:
            r = 0
        if r > 255:
            r = 255

        return ColorClass(int(r), int(g), int(b))

    def __add__(self, other):
        return ColorClass(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b)

    def toBytes(self):

        if self.b > 255:
            self.b = 255
        if self.g > 255:
            self.g = 255
        if self.r > 255:
            self.r = 255

        if self.b < 0:
            self.b = 0
        if self.g < 0:
            self.g = 0
        if self.r < 0:
            self.r = 0

        return bytes([self.b, self.g, self.r])

    def ___repr__(self):
        return "color(%s, %s, %s)" % (self.r, self.g, self.b)

class EnvmapClass():
    def __init__(self, path):
        self.path = path
        self.read()

    def read(self):
        with open(self.path, "rb") as image:
            image.seek(10)
            header_size = struct.unpack("=l", image.read(4))[0]
            image.seek(18)
            self.width = struct.unpack("=l", image.read(4))[0]
            self.height = struct.unpack("=l", image.read(4))[0]

            image.seek(header_size)

            self.pixels = []
            for y in range(self.height):
                self.pixels.append([])
                for x in range(self.width):
                    b = ord(image.read(1))
                    g = ord(image.read(1))
                    r = ord(image.read(1))
                    self.pixels[y].append(
                        ColorClass(r, g, b)
                    )

    def get_color(self, direction):
        direction = direction.normalize()
        x = int(((atan2(direction.z, direction.x) / (2 * pi)) + 0.5) * self.width)
        y = int((acos(direction.y) / pi) * self.height)

        return self.pixels[-y][x]

class LightClass:
    def __init__(self, position, intensity, color):
        self.position = position
        self.intensity = intensity
        self.color = color

class TextureClass():
    def __init__(self, path):
        self.path = path
        self.read()

    def read(self):
        with open(self.path, "rb") as image:
            image.seek(10)
            header_size = struct.unpack("=l", image.read(4))[0]
            image.seek(18)
            self.width = struct.unpack("=l", image.read(4))[0]
            self.height = struct.unpack("=l", image.read(4))[0]

            image.seek(header_size)

            self.pixels = []
            for y in range(self.height):
                self.pixels.append([])
                for x in range(self.width):
                    b = ord(image.read(1))
                    g = ord(image.read(1))
                    r = ord(image.read(1))
                    self.pixels[y].append(
                        ColorClass(r, g, b)
                    )

    def getColor(self, tx, ty):
        x = round(tx * (self.width)-1)
        y = round(ty * (self.height)-1)

        try:
            return self.pixels[y][x]
        except:
            x = int(max(min(x, self.width), 0))
            y = int(max(min(x, self.height), 0))
            return self.pixels[y][x]

class MaterialClass:
    def __init__(self, diffuse, albedo, spec, refractive_index=0, textura=None):
        self.refractive_index = refractive_index
        self.diffuse = diffuse
        self.albedo = albedo
        self.spec = spec
        if textura:
            self.textura = TextureClass(textura)
        else:
            self.textura = None

class Vector3Class():
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
    def round(self):
        self.x = round(self.x)
        self.y = round(self.y)
        self.z = round(self.z)
    def __repr__(self):
        return "V3(%s, %s, %s)" % (self.x, self.y, self.z)
    def __mul__(self, other):
        if(type(other) == int or type(other) == float):
            return Vector3Class(
                self.x * other,
                self.y * other,
                self.z * other
            )
        return Vector3Class(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    def __add__(self, other):
        return Vector3Class(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    def __sub__(self, other):
        return Vector3Class(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    def cross(self, other):
        return Vector3Class(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    def normalize(self):
        try:
            return self * (1/self.length())
        except:
            return Vector3Class(0, 0, 0)

class PlaneObject():
    def __init__(self, center, w, l, material):
        self.center = center
        self.w = w
        self.l = l
        self.material = material

    def ray_intersect(self, origin, direction):
        d = -(origin.y + self.center.y) / direction.y
        impact = (direction * d) - origin
        normal = Vector3Class(0, 1, 0)

        if d <= 0 or \
                impact.x > (self.center.x + self.w/2) or impact.x < (self.center.x - self.w/2) or \
                impact.z > (self.center.z + self.l/2) or impact.z < (self.center.z - self.l/2):
            return None

        return IntersectClass(
            distance=d,
            point=impact,
            normal=normal
        )

class writeUtilsClass():
    def char(c):
        return struct.pack("=c", c.encode('ascii'))

    def word(w):
        return struct.pack("=h", w)

    def dword(d):
        return struct.pack("=l", d)

class libClass():
    def reflect(I, N):
        return (I - N * 2 * (I @ N)).normalize()

    def refract(I, N, roi):
        etai = 1
        etat = roi

        cosi = (I @ N) * -1

        if(cosi < 0):
            cosi *= -1
            etai *= -1
            etat *= -1
            N *= -1

        eta = etai/etat

        k = 1 - eta**2 * (1 - cosi**2)

        if k < 0:
            return Vector3Class(0, 0, 0)

        cost = k ** 0.5

        return ((I * eta) + (N * (eta * cosi - cost))).normalize()

    def writeBMP(filename, width, height, framebuffer):
        f = open(filename, 'bw')
        f.write(writeUtilsClass.char('B'))
        f.write(writeUtilsClass.char('M'))
        f.write(writeUtilsClass.dword(14 + 40 + width * height * 3))
        f.write(writeUtilsClass.word(0))
        f.write(writeUtilsClass.word(0))
        f.write(writeUtilsClass.dword(14 + 40))
        f.write(writeUtilsClass.dword(40))
        f.write(writeUtilsClass.dword(width))
        f.write(writeUtilsClass.dword(height))
        f.write(writeUtilsClass.word(1))
        f.write(writeUtilsClass.word(24))
        f.write(writeUtilsClass.dword(0))
        f.write(writeUtilsClass.dword(width * height * 3))
        f.write(writeUtilsClass.dword(0))
        f.write(writeUtilsClass.dword(0))
        f.write(writeUtilsClass.dword(0))
        f.write(writeUtilsClass.dword(0))
        for y in range(height):
            for x in range(width):
                f.write(framebuffer[y][x])
        f.close()    

class RaytracerClass():

    MAX_RECURSION_DEPTH = 3

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.background_color = ColorClass(0, 0, 255)
        self.current_color = ColorClass(255, 255, 255)
        self.clear()
        self.scene = []
        self.density = 1

    def clear(self):
        self.framebuffer = [
            [self.background_color for x in range(self.width)]
            for y in range(self.height)
        ]

    def point(self, x, y, c=None):
        if y >= 0 and y < self.height and x >= 0 and x < self.width:
            self.framebuffer[y][x] = c.toBytes(
            ) or self.current_color.toBytes()

    def write(self, filename):
        libClass.writeBMP(filename, self.width, self.height, self.framebuffer)

    def render(self):
        with Bar('Renderizando...', max=self.height) as bar:
            fov = int(pi/2)
            ar = self.width/self.height
            tana = tan(fov/2)
            for y in range(self.height):
                for x in range(self.width):
                    if random.random() < self.density:
                        i = ((2 * (x + 0.5) / self.width) - 1) * ar * tana
                        j = (1 - (2 * (y + 0.5) / self.height)) * tana

                        direction = Vector3Class(i, j, -1).normalize()
                        origin = Vector3Class(0, 0, 0)
                        c = self.cast_ray(origin, direction)

                        self.point(x, y, c)
                bar.next()  

    def get_background(self, direction):
        if self.envmap:
            return self.envmap.get_color(direction)
        else:
            return self.background_color

    def cast_ray(self, origin, direction, recursion=0):

        if recursion >= self.MAX_RECURSION_DEPTH:
            return self.get_background(direction)

        material, intersect = self.scene_intersect(origin, direction)

        if material is None:
            return self.get_background(direction)

        if material.textura:
            material.diffuse = material.textura.getColor(
                intersect.coords.x, intersect.coords.y)

        light_dir = (self.light.position - intersect.point).normalize()

        if material.albedo[2] > 0:
            reverse_direction = direction * -1
            reflect_direction = libClass.reflect(
                reverse_direction, intersect.normal)
            reflect_bias = -0.5 if reflect_direction @ intersect.normal < 0 else 0.5
            reflect_origin = intersect.point + \
                (intersect.normal * reflect_bias)
            reflect_color = self.cast_ray(
                reflect_origin, reflect_direction, recursion + 1)
        else:
            reflect_color = ColorClass(0, 0, 0)

        reflection = reflect_color * material.albedo[2]

        if material.albedo[3] > 0:
            refract_direction = libClass.refract(
                direction, intersect.normal, material.refractive_index)
            refract_bias = -0.5 if refract_direction @ intersect.normal < 0 else 0.5
            refract_origin = intersect.point + \
                (intersect.normal * refract_bias)
            refract_color = self.cast_ray(
                refract_origin, refract_direction, recursion + 1)
        else:
            refract_color = ColorClass(0, 0, 0)

        refraction = refract_color * material.albedo[3]

        shadow_bias = 1.1
        shadow_origin = intersect.point + (intersect.normal * shadow_bias)
        shadow_material, shadow_intersect = self.scene_intersect(
            shadow_origin, light_dir)

        shadow_intensity = 0
        if shadow_material:
            shadow_intensity = 0.7

        dIntensity = light_dir @ intersect.normal
        diffuse = material.diffuse * dIntensity * \
            material.albedo[0] * \
            (1 - shadow_intensity)

        light_dir = libClass.reflect(light_dir, intersect.normal)
        rIntensity = max(0, light_dir @ direction)
        specularI = self.light.intensity * rIntensity ** material.spec
        specular = self.light.color * specularI * material.albedo[1]

        return diffuse + specular + reflection + refraction

    def scene_intersect(self, origin, direction):
        zbuffer = 99999
        material = None
        intersect = None

        for o in self.scene:
            object_intersect = o.ray_intersect(origin, direction)
            if object_intersect:
                if object_intersect.distance < zbuffer:
                    zbuffer = object_intersect.distance
                    material = o.material
                    intersect = object_intersect
        return material, intersect

class CubeFractionClass():
    def __init__(self, position, normal,  material):
        self.position = position
        self.normal = normal.normalize()
        self.material = material

    def ray_intersect(self, orig, dir):
        denom = dir @ self.normal
        if abs(denom) > 0.0001:
            t = (self.normal @ (self.position - orig)) / denom
            if t > 0:
                hit = orig + (dir * t)
                return IntersectClass(
                    distance=t,
                    point=hit,
                    normal=self.normal,
                )
        return None

class CubeObject():
    def __init__(self, position, size, material):
        self.position = position
        self.size = size
        self.material = material
        self.planes = []

        halfSize = size / 2

        self.planes.append(
            CubeFractionClass((position + Vector3Class(halfSize, 0, 0)), Vector3Class(1, 0, 0), material))
        self.planes.append(
            CubeFractionClass((position + Vector3Class(-halfSize, 0, 0)), Vector3Class(-1, 0, 0), material))

        self.planes.append(
            CubeFractionClass((position + Vector3Class(0, halfSize, 0)), Vector3Class(0, 1, 0), material))
        self.planes.append(
            CubeFractionClass((position + Vector3Class(0, -halfSize, 0)), Vector3Class(0, -1, 0), material))

        self.planes.append(
            CubeFractionClass((position + Vector3Class(0, 0, halfSize)), Vector3Class(0, 0, 1), material))
        self.planes.append(
            CubeFractionClass((position + Vector3Class(0, 0, -halfSize)), Vector3Class(0, 0, -1), material))

    def ray_intersect(self, orig, direction):

        epsilon = 0.001

        boundsMin = [0, 0, 0]
        boundsMax = [0, 0, 0]

        boundsMin[0] = self.position.x - (epsilon + self.size / 2)
        boundsMax[0] = self.position.x + (epsilon + self.size / 2)
        boundsMin[1] = self.position.y - (epsilon + self.size / 2)
        boundsMax[1] = self.position.y + (epsilon + self.size / 2)
        boundsMin[2] = self.position.z - (epsilon + self.size / 2)
        boundsMax[2] = self.position.z + (epsilon + self.size / 2)

        t = float('inf')
        intersect = None

        for plane in self.planes:

            planeInter = plane.ray_intersect(orig, direction)

            if planeInter is not None:

                if planeInter.point.x >= boundsMin[0] and planeInter.point.x <= boundsMax[0]:
                    if planeInter.point.y >= boundsMin[1] and planeInter.point.y <= boundsMax[1]:
                        if planeInter.point.z >= boundsMin[2] and planeInter.point.z <= boundsMax[2]:
                            if planeInter.distance < t:
                                t = planeInter.distance
                                intersect = planeInter

        if intersect is None:
            return None

        x, y = self.getNormal(0, intersect.point)

        return IntersectClass(
            distance=intersect.distance,
            point=intersect.point,
            normal=intersect.normal,
            coords=Vector3Class(x, y, 0)
        )

    def getNormal(self, face, impact):
        if face == 0:
            minH = (self.position.z - self.size/2)
            minV = (self.position.y - self.size/2)

            z = (impact.z - minH) / self.size
            y = (impact.y - minV) / self.size

            return z, y

        elif face == 1:
            minH = (self.position.z + self.size/2)
            minV = (self.position.y - self.size/2)

            z = (impact.z - minH) / self.size
            y = (impact.y - minV) / self.size

            return z, y

        elif face == 2:
            minH = (self.position.x - self.size/2)
            minV = (self.position.z - self.size/2)

            x = (impact.x - minH) / self.size
            z = (impact.z - minV) / self.size

            return x, z

        elif face == 3:

            minH = (self.position.x - self.size/2)
            minV = (self.position.z + self.size/2)

            x = (impact.x - minH) / self.size
            z = (impact.z - minV) / self.size

            return x, z

        elif face == 4:
            minH = (self.position.x - self.size/2)
            minV = (self.position.y - self.size/2)

            x = (impact.x - minH) / self.size
            y = (impact.y - minV) / self.size

            return x, y

        elif face == 5:
            minH = (self.position.x + self.size/2)
            minV = (self.position.y - self.size/2)

            x = (impact.x - minH) / self.size
            y = (impact.y - minV) / self.size

            return x, y

class Proyecto2RaytracerClass():

    def inicializar(self):

        canvas = RaytracerClass(int(sys.argv[1]), int(sys.argv[2]))
        
        mVerdeClaro   = MaterialClass(diffuse=ColorClass(159, 229, 0), albedo=[0.9, 0.2, 0, 0], spec=100)
        mVerdeOscuro  = MaterialClass(diffuse=ColorClass(82, 95, 49), albedo=[0.9, 0.2, 0, 0], spec=100)
        mNegro        = MaterialClass(diffuse=ColorClass(0, 0, 0),albedo=[0, 0, 0, 0], spec=100)
        mNegroMetal   = MaterialClass(diffuse=ColorClass(100, 100, 100),albedo=[0.7, 0.3, 0.0, 0.0], spec=10)
        mReflectivo   = MaterialClass(diffuse=ColorClass(255, 255, 255),albedo=[0, 1, 0.8, 0], spec=1430)


        posicionLuz   = Vector3Class(-20, 20, 20)
        intensidadLuz = 1.5
        colorLuz      = ColorClass(255, 255, 255) # Color blanco
        canvas.light  = LightClass(posicionLuz, intensidadLuz, colorLuz)

        ref_centro = 2.0

        canvas.scene = [

            # Suelo:
            PlaneObject(Vector3Class(0.0, -2.4, -5), 1.5, 1.5, mNegro), 
            PlaneObject(Vector3Class(1.5, -2.4, -5), 1.5, 1.5, mNegro),
            PlaneObject(Vector3Class(-1.5, -2.4, -5), 1.5, 1.5, mReflectivo),
            PlaneObject(Vector3Class(0.0, -2.4, -6.5), 1.5, 1.5, mReflectivo), 
            PlaneObject(Vector3Class(1.5, -2.4, -6.5), 1.5, 1.5, mReflectivo),
            PlaneObject(Vector3Class(-1.5, -2.4, -6.5), 1.5, 1.5, mNegro),
            PlaneObject(Vector3Class(0.0, -2.4, -8), 1.5, 1.5, mNegro), 
            PlaneObject(Vector3Class(1.5, -2.4, -8), 1.5, 1.5, mNegro),
            PlaneObject(Vector3Class(-1.5, -2.4, -8), 1.5, 1.5, mReflectivo),

            # Piernas:
            CubeObject(Vector3Class(0, ref_centro -0.5  , -5), 0.5, mVerdeOscuro),
            CubeObject(Vector3Class(0, ref_centro -1    , -5), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(1, ref_centro -0.5  , -5), 0.5, mVerdeOscuro),
            CubeObject(Vector3Class(1, ref_centro -1    , -5), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0.5, ref_centro -1    , -5.5), 0.5, mVerdeClaro),


            # Cuerpo 3:
            CubeObject(Vector3Class(1, ref_centro -1.5  , -5.5), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0, ref_centro -1.5  , -5.5), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0.5, ref_centro -1.5  , -5.5), 0.5, mVerdeOscuro),
            #Cuerpo 2:
            CubeObject(Vector3Class(1, ref_centro -2    , -5.5), 0.5, mVerdeOscuro),
            CubeObject(Vector3Class(0, ref_centro -2    , -5.5), 0.5, mVerdeOscuro),
            CubeObject(Vector3Class(0.5, ref_centro -2  , -5.5), 0.5, mVerdeClaro),
            #Cuerpo 1:
            CubeObject(Vector3Class(1, ref_centro -2.5    , -5.5), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0, ref_centro -2.5    , -5.5), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0.5, ref_centro -2.5  , -5.5), 0.5, mVerdeOscuro),

            # Cabeza 3:
            CubeObject(Vector3Class(1, ref_centro -3    , -5.0), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0, ref_centro -3    , -5.0), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0.5, ref_centro -3  , -5.0), 0.5, mNegro),
            # Cabeza 2:
            CubeObject(Vector3Class(1, ref_centro -3.5    , -5.0), 0.5, mNegroMetal),
            CubeObject(Vector3Class(0, ref_centro -3.5    , -5.0), 0.5, mNegroMetal),
            CubeObject(Vector3Class(0.5, ref_centro -3.5  , -5.0), 0.5, mVerdeClaro),
            # Cabeza 1:
            CubeObject(Vector3Class(1, ref_centro -4    , -5.0), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0, ref_centro -4    , -5.0), 0.5, mVerdeClaro),
            CubeObject(Vector3Class(0.5, ref_centro -4  , -5.0), 0.5, mVerdeOscuro),

        ]

        canvas.envmap = EnvmapClass("./assets/bgdark.bmp")
        canvas.render()
        canvas.write("./build/resultado.bmp")
    
if __name__ == "__main__":
    proyecto = Proyecto2RaytracerClass()
    proyecto.inicializar()



SUMMARY = "Ajout modeles"
LICENSE = "CLOSED"

S = "${WORKDIR}"

SRC_URI = "https://github.com/yassinthabet/projet_PFE/raw/main/main.py;name=main \ 
          https://github.com/yassinthabet/projet_PFE/raw/main/marque.json;name=marque \
          https://github.com/yassinthabet/projet_PFE/raw/main/config.py;name=config \
          https://github.com/yassinthabet/projet_PFE/raw/main/utils.py;name=utils \
          https://github.com/yassinthabet/projet_PFE/raw/main/classes.names;name=classes \
          https://github.com/yassinthabet/projet_PFE/raw/main/coco.names;name=coco \
          file://darknet-yolov3.cfg \
          file://demo.mp4 \
          file://model.weights \
          file://Nationality.pth \
          file://yolov4.cfg \
          file://yolov4.weights \
          file://brand.pth \
          file://video1.mp4 \
          file://47.mp4"
          

SRC_URI[main.sha256sum] = "d26a4314a4aa6b3961a173fd81213794b75996f1cd55abecaa3e3a69836320cf"
SRC_URI[marque.sha256sum] = "a0df51b79b30173515ddd01f700121906d225eeb69d67e45119d3935c15e5e40"
SRC_URI[config.sha256sum] = "67821a3be502fd88ddbe48e30670bad810652e4f96ad262ea79588f60cae9682"
SRC_URI[utils.sha256sum] = "985466292d10e620ccf36a977f1327c5d844691e99dd745945b93c812733f8d6"
SRC_URI[classes.sha256sum] = "327d67f372f345b3cc84d142a70a06173a1cd0cb0208a2a3bf9eccec89886032"
SRC_URI[coco.sha256sum] = "33c77761e124cc74911346865e3bc1219b87c2db7d0f106e3376bf5ef3785933"

FILES:${PN} += "${datadir}/application/*"

do_install() {
    install -d ${D}${datadir}/application/ 
    install -m 0644 ${WORKDIR}/classes.names ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/coco.names ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/config.py ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/darknet-yolov3.cfg ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/demo.mp4 ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/main.py ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/model.weights ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/Nationality.pth ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/utils.py ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/yolov4.cfg ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/yolov4.weights ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/brand.pth ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/marque.json ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/video1.mp4 ${D}${datadir}/application/
    install -m 0644 ${WORKDIR}/47.mp4 ${D}${datadir}/application/
}




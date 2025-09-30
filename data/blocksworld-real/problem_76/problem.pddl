(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        blue - block
        green - block
        purple - block
        yellow - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on blue red)
        (on green blue)
        (on purple green)
        (on yellow purple)
        (on orange yellow)
        (clear orange)
        ; red -> blue -> green -> purple -> yellow -> orange
    )
    (:goal (and (on red purple)
        (on orange blue)
        (on green orange)
        (on yellow green)))
)
    
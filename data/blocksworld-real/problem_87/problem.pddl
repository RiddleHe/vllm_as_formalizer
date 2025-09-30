(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        green - block
        yellow - block
        purple - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on green blue)
        (on yellow green)
        (on purple yellow)
        (clear purple)
        (ontable red)
        (on orange red)
        (clear orange)
        ; blue -> green -> yellow -> purple
        ; red -> orange
    )
    (:goal (and (on green red)
        (on purple blue)
        (on yellow orange)))
)
    
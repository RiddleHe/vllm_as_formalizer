(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        green - block
        blue - block
        purple - block
        yellow - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on green red)
        (on blue green)
        (on purple blue)
        (on yellow purple)
        (clear yellow)
        (ontable orange)
        (clear orange)
        ; red -> green -> blue -> purple -> yellow
        ; orange
    )
    (:goal (and (on green red)
        (on purple yellow)
        (on blue orange)))
)
    
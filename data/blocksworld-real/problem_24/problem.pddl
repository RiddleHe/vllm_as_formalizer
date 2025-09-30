(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        blue - block
        green - block
        purple - block
        red - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable blue)
        (on green blue)
        (on purple green)
        (clear purple)
        (ontable red)
        (on yellow red)
        (clear yellow)
        ; orange
        ; blue -> green -> purple
        ; red -> yellow
    )
    (:goal (and (on red yellow)
        (on purple red)
        (on green purple)
        (on orange green)
        (on blue orange)))
)
    
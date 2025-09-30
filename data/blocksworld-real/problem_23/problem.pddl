(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        purple - block
        yellow - block
        green - block
        blue - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on purple red)
        (clear purple)
        (ontable yellow)
        (on green yellow)
        (on blue green)
        (clear blue)
        (ontable orange)
        (clear orange)
        ; red -> purple
        ; yellow -> green -> blue
        ; orange
    )
    (:goal (and (on yellow blue)
        (on green purple)
        (on orange green)
        (on red orange)))
)
    
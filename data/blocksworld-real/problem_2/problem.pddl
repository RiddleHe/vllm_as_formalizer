(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        purple - block
        blue - block
        orange - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on purple red)
        (on blue purple)
        (on orange blue)
        (clear orange)
        (ontable yellow)
        (on green yellow)
        (clear green)
        ; red -> purple -> blue -> orange
        ; yellow -> green
    )
    (:goal (and (on orange yellow)
        (on blue orange)
        (on purple blue)
        (on red purple)
        (on green red)))
)
    